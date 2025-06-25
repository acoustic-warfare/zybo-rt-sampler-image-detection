import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # Ensure you have the SORT tracker installed: pip install sort


class yolo_model:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def get_detections(self, frame, conf_threshold=0.0):
        results = self.model.predict(source=frame, stream=False, verbose=False)
        all_boxes = []
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                conf = boxes.conf[i].item()
                if conf >= conf_threshold:  
                    all_boxes.append([*xyxy, conf])
        return all_boxes


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def extract_patch(frame, box, scale=1.2):
    x1, y1, x2, y2 = map(int, box)
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2
    nw, nh = int(w * scale), int(h * scale)
    nx1 = max(0, cx - nw // 2)
    ny1 = max(0, cy - nh // 2)
    nx2 = min(frame.shape[1], cx + nw // 2)
    ny2 = min(frame.shape[0], cy + nh // 2)
    return frame[ny1:ny2, nx1:nx2]


def cross_correlation_score(prev_patch, curr_patch):
    if prev_patch.shape != curr_patch.shape:
        curr_patch = cv2.resize(curr_patch, (prev_patch.shape[1], prev_patch.shape[0]))
    result = cv2.matchTemplate(curr_patch, prev_patch, cv2.TM_CCOEFF_NORMED)
    return np.max(result)


def track_with_correlation(prev_frame, curr_frame, prev_box):
    prev_patch = extract_patch(prev_frame, prev_box)
    search_area = extract_patch(curr_frame, prev_box, scale=1.5)
    result = cv2.matchTemplate(search_area, prev_patch, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    dx, dy = max_loc
    new_x1 = prev_box[0] + dx
    new_y1 = prev_box[1] + dy
    new_x2 = prev_box[2] + dx
    new_y2 = prev_box[3] + dy
    return [new_x1, new_y1, new_x2, new_y2], max_val


def process_video(frame_queue, output_queue, stream=False, show=False, model_path=None):
    detector = yolo_model(model_path)
    confh = 0.5
    confl = 0.1
    iou_thresh = 0.5
    corr_thresh = 0.8

    prev_frame = None
    prev_detections = []

    while True:
        try:
            frame_number, frame = frame_queue.get()
            frame_queue.task_done()
        except Exception as e:
            print("No frame received:", e)
            continue

        try:
            detections = detector.get_detections(frame, conf_threshold=confl)
            valid = [d for d in detections if d[4] > confh]
            candidates = [d for d in detections if confl < d[4] <= confh]

            if valid:
                prev_detections = valid
                prev_frame = frame.copy()
                for box in valid:
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = box[4]
                    label = f"{conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                if show:
                    cv2.imshow("Frame", frame)
                    if cv2.waitKey(1) == 27:
                        break
                output_queue.put((frame_number, frame))
                continue

            # If no valid detections, process candidates
            for cand in candidates:
                for prev in prev_detections:
                    pred_box, corr_score = track_with_correlation(
                        prev_frame, frame, prev[:4]
                    )
                    iou = compute_iou(pred_box, cand[:4])
                    if iou > iou_thresh or corr_score > corr_thresh:
                        cand[4] = confh  # Boost confidence
                        break
                else:
                    cand[4] = 0  # Consider lost

            prev_detections = [d for d in detections if d[4] >= confh]
            prev_frame = frame.copy()
            # Draw detections
            for box in prev_detections:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )
            if show:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) == 27:
                    break
            output_queue.put((frame_number, frame))

        except Exception as e:
            print(f"YOLO tracking error: {e}")
            output_queue.put((frame_number, frame))

def process_video_boxes_only(frame_queue, output_queue, stream=False, show=False, model_path=None):
    detector = yolo_model(model_path)
    confh = 0.7
    confl = 0.3
    iou_thresh = 0.5
    corr_thresh = 0.8

    prev_frame = None
    prev_detections = []

    while True:
        try:
            frame_number, frame = frame_queue.get()
            frame_queue.task_done()
        except Exception as e:
            print("No frame received:", e)
            continue

        try:
            # Create a blank image with the same shape as the input frame
            blank = np.zeros_like(frame)

            detections = detector.get_detections(frame, conf_threshold=confl)
            valid = [d for d in detections if d[4] > confh]
            candidates = [d for d in detections if confl < d[4] <= confh]

            if valid:
                prev_detections = valid
                prev_frame = frame.copy()
                for box in valid:
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = box[4]
                    label = f"{conf:.2f}"
                    cv2.rectangle(blank, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        blank, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                if show:
                    cv2.imshow("Boxes Only", blank)
                    if cv2.waitKey(1) == 27:
                        break
                output_queue.put((frame_number, blank))
                continue

            # If no valid detections, process candidates
            for cand in candidates:
                for prev in prev_detections:
                    pred_box, corr_score = track_with_correlation(
                        prev_frame, frame, prev[:4]
                    )
                    iou = compute_iou(pred_box, cand[:4])
                    if iou > iou_thresh or corr_score > corr_thresh:
                        cand[4] = confh  # Boost confidence
                        break
                else:
                    cand[4] = 0  # Consider lost

            prev_detections = [d for d in detections if d[4] >= confh]
            prev_frame = frame.copy()
            # Draw detections
            for box in prev_detections:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                cv2.rectangle(blank, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{conf:.2f}"
                cv2.putText(
                    blank, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )
            if show:
                cv2.imshow("Boxes Only", blank)
                if cv2.waitKey(1) == 27:
                    break
            output_queue.put((frame_number, blank))

        except Exception as e:
            print(f"YOLO tracking error: {e}")
            output_queue.put((frame_number, blank))


def process_video_track(video_path, model_path, rec=True):
    detector = yolo_model(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = Sort()  # Initialize SORT tracker

    if rec:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output3.mp4", fourcc, fps, (width, height))

    confh = 0.7
    confl = 0.3
    iou_thresh = 0.5
    corr_thresh = 0.8

    prev_frame = None
    prev_detections = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 2 != 0:
            continue

        detections = detector.get_detections(frame, conf_threshold=confl)
        valid = [d for d in detections if d[4] > confh]
        candidates = [d for d in detections if confl < d[4] <= confh]

        # Prepare detections for SORT: [x1, y1, x2, y2, conf]
        dets = np.array(valid + candidates)
        if len(dets) == 0:
            dets = np.empty((0, 5))

        # Update tracker and get tracks: [x1, y1, x2, y2, track_id]
        tracks = tracker.update(dets)

        # Draw tracked boxes with IDs
        for track in tracks:
            x1, y1, x2, y2, track_id = track.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        # Optionally, fallback to your correlation logic if no valid detections
        if len(valid) == 0 and len(candidates) > 0 and prev_frame is not None:
            for cand in candidates:
                for prev in prev_detections:
                    pred_box, corr_score = track_with_correlation(
                        prev_frame, frame, prev[:4]
                    )
                    iou = compute_iou(pred_box, cand[:4])
                    if iou > iou_thresh or corr_score > corr_thresh:
                        cand[4] = confh  # Boost confidence
                        break
                else:
                    cand[4] = 0  # Consider lost

        prev_detections = [d for d in detections if d[4] >= confh]
        prev_frame = frame.copy()

        if rec:
            out.write(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break

    if rec:
        out.release()
    cap.release()
    cv2.destroyAllWindows()




if __name__ == "__main__":

    process_video(
        "../test_footage/video/flera_dron.mp4",
        "runs/detect/train4/weights/best_of_all.pt",
    )

import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import (
    Sort,
)  # Ensure you have the SORT tracker installed: pip install sort


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


def process_video(video_path, model_path, rec=True):
    detector = yolo_model(model_path)
    cap = cv2.VideoCapture(video_path)

    if rec:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = 25

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output4.mp4", fourcc, fps, (width, height))

    confh = 0.7
    confl = 0.3
    iou_thresh = 0.5
    corr_thresh = 0.8

    prev_frame = None
    prev_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.get_detections(frame, conf_threshold=confl)
        valid = [d for d in detections if d[4] > confh]
        candidates = [d for d in detections if confl < d[4] <= confh]

        if valid:
            prev_detections = valid
            prev_frame = frame.copy()
            for box in valid:
                x1, y1, x2, y2 = map(int, box[:4])
                conf = box[4]
                # Draw bounding box with label and confidence
                label = f"{conf:.2f}"
                print(f"Confidence: {conf:.2f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            if rec:
                out.write(frame)  # Write the frame with detections to output video
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:
                break
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
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
        if rec:
            out.write(frame)  # Write the frame with detections to output video
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == 27:
            break
            break

    cap.release()
    if rec:
        out.release()
    cv2.destroyAllWindows()


def process_video_track(video_path, model_path, rec=True):
    detector = yolo_model(model_path)
    cap = cv2.VideoCapture(video_path)
    tracker = Sort()  # Initialize SORT tracker

    if rec:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        if fps == 0:
            fps = 30

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output3.mp4", fourcc, fps, (width, height))
    else:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

    delay = int(1000 / fps)  # Calculate delay for cv2.waitKey

    confh = 0.65
    confl = 0.3
    iou_thresh = 0.5
    corr_thresh = 0.8

    prev_frame = None
    prev_detections = []
    # frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            
            conf = 0
            # Find the detection with highest IoU to this track to get its confidence
            for det in dets:
                iou = compute_iou([x1, y1, x2, y2], det[:4])
                if iou > 0.5:
                    conf = det[4]
                    break
            label = f"Conf:{conf:.2f}"
            cv2.putText(
                frame,
                label,
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
        if cv2.waitKey(delay) == 27:
            break

    if rec:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

from multiprocessing import JoinableQueue, Value
def process_video_track_boxes_only(frame_queue, output_queue, stream=False, show=False, model_path=None, running: Value = None):
    detector = yolo_model(model_path)
    tracker = Sort()  # SORT tracker
    confh = 0.7
    confl = 0.4
    iou_thresh = 0.5
    corr_thresh = 0.8
    rectangle_coords_conf = [[0, 0, 0, 0, 0, 0]]

    prev_frame = None
    prev_detections = []

    while running:
        try:
            frame_number, frame = frame_queue.get()
            frame_queue.task_done()
        except Exception as e:
            print("No frame received:", e)
            continue

        try:
            # Ensure frame is color
            blank = np.zeros_like(frame)
            if len(blank.shape) == 2:
                blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

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
            coords_index = 0
            for track in tracks:
                x1, y1, x2, y2, track_id = track.astype(int)
                cv2.rectangle(blank, (x1, y1), (x2, y2), (0, 255, 0), 2)
                conf = 0
                # Find the detection with highest IoU to this track to get its confidence
                for det in dets:
                    iou = compute_iou([x1, y1, x2, y2], det[:4])
                    if iou > 0.5:
                        conf = det[4]
                        break
                label = f"ID:{int(track_id)} Conf:{conf:.2f}"
                cv2.putText(
                    blank, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
                if coords_index < len(rectangle_coords_conf):
                    rectangle_coords_conf[coords_index][0] = x1
                    rectangle_coords_conf[coords_index][1] = y1
                    rectangle_coords_conf[coords_index][2] = x2
                    rectangle_coords_conf[coords_index][3] = y2
                    rectangle_coords_conf[coords_index][4] = conf
                    rectangle_coords_conf[coords_index][5] = track_id
                    
                else:
                    rectangle_coords_conf.append([x1, y1, x2, y2, conf, track_id])
                coords_index += 1

            if show:
                cv2.imshow("Boxes Only", blank)
                if cv2.waitKey(1) == 27:
                    break
            if output_queue.full():
                try:
                    output_queue.get_nowait()  # Remove old frame
                except:
                    pass
            output_queue.put((frame_number, blank, rectangle_coords_conf))

        except Exception as e:
            print(f"YOLO tracking error: {e}")
            output_queue.put((frame_number, blank, [0, 0, 0, 0, 0, 0]))
        except KeyboardInterrupt:
            print("Keyboard interrupt received, stopping tracking.")
            if running is not None:
                running.value = False
            break


if __name__ == "__main__":

    process_video_track(
        "/dev/video2",
        "runs/detect/train4/weights/best_of_all.pt",
        rec=False,
    )

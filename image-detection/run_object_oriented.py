import time
from ultralytics import settings,YOLO
class yolo_model:
    
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def train(self, data_path, epochs=5, imgsz=320):
        self.data_path = data_path
        results = self.model.train(data=self.data_path, epochs=epochs, imgsz=imgsz)
        return results
    
    def run_inference(self, source, stream=False, show=True):
        results = self.model(source=source, stream=stream, show=show)
        return results

    def print_confidences(self, results):
        for result in results:
            confidences = result.boxes.conf.cpu().numpy()
            for conf in confidences:
                print(f"Confidence: {conf:.2f}")
    
    def run_conf_n_inference(self, frame_queue, output_queue, stream=False, show=False):
        while True:
            try:
                frame = frame_queue.get()  # get frame from camera_reader
                frame_queue.task_done()  # Mark frame as processed
                print(f"YOLO: Got frame with shape {frame.shape}")  # Debug print
            except Exception as e:
                print("No frame received:", e)
                continue
                
            try:
                results = self.run_inference(source=frame, stream=stream, show=show)
                
                # Process results and draw bounding boxes
                for result in results:
                    # Get the annotated frame with bounding boxes
                    output_frame = result.plot()
                    print("YOLO: putting frame in queue")
                    output_queue.put(output_frame)
                    break  # Only process first result for single frame
                    
            except Exception as e:
                print(f"YOLO inference error: {e}")
                # Put original frame if inference fails
                output_queue.put(frame)
     
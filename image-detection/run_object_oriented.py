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
    
    def run_conf_n_inference(self, source, stream=False, show=True):
        for result in self.model(source=source, stream=stream, show=show):
            confidences = result.boxes.conf.cpu().numpy()   
            for conf in confidences:
                print(f"Confidence: {conf:.2f}")
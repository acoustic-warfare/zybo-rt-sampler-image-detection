import cv2
import numpy as np
class sensorfusiondecider:
    def __init__(self):
        self.hej = 0
        
    def get_lightlevel(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_brightness = cv2.mean(gray_image)[0]
        return average_brightness/255.0  # Normalize to [0, 1] range
    
    def get_entropy(self, heatmap):
        s = np.sum(heatmap)
        if s > 0:
            heatmap = heatmap / s
        else:
            heatmap = np.zeros_like(heatmap)
        entropy = -np.sum(heatmap * np.log(heatmap + 1e-12))
        confidence = 1 / (1 + entropy)  
        return confidence
        
    def create_image(self, image, yolo_image, power_image, heatmap):
        display_size = (640, 360)

        # Ensure all images are 3-channel and same size
        def ensure_shape(img):
            img = cv2.resize(img, display_size)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.dtype != np.uint8:
                img = (255 * np.clip(img, 0, 1)).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
            return img

        image = ensure_shape(image)
        yolo_image = ensure_shape(yolo_image)
        power_image = ensure_shape(power_image)
        heatmap = ensure_shape(heatmap)

        image, yolo_image, power_image = self.get_decision(image, yolo_image, power_image, heatmap)

        yolo_image = cv2.flip(yolo_image, 1)
        combined = cv2.addWeighted(image, 1, yolo_image, 0.7, 0)
        combined = cv2.addWeighted(combined, 1, power_image, 0.7, 0)
        combined = cv2.addWeighted(combined, 1, heatmap, 0.7, 0)

        combined_resized = cv2.flip(combined, 1)  # Flip combined image

        return combined_resized
    
    def get_decision(self, image, yolo_image, power_image, heatmap):
        #Start by checking the light level
        light_level = self.get_lightlevel(image)    
        blank = np.zeros_like(image)
        print(light_level)
        if light_level < 0.2:
            print("Low light level, using heatmap")
            yolo_image = blank

        #Now check the heatmap for entropy (multiple sources of sound or not)
        entropy_conf = self.get_entropy(heatmap)
        print("Entropy confidence:", entropy_conf)


        
        return image, yolo_image, power_image


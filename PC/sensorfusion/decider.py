import cv2
import numpy as np
class sensorfusiondecider:
    def __init__(self, display_size=(640, 360), MAX_ANGLE=30, ASPECT_RATIO=16/9):
        self.display_size = display_size
        self.image_confidence_threshold = 0.5
        self.MAX_X = MAX_ANGLE
        self.MAX_Y = MAX_ANGLE / ASPECT_RATIO
        
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
        
    def create_image(self, image, yolo_image,power_detection_img, yolo_rect_conf, power_rect, heatmap):

        # Ensure all images are 3-channel and same size
        def ensure_shape(img):
            img = cv2.resize(img, self.display_size)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.dtype != np.uint8:
                img = (255 * np.clip(img, 0, 1)).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
            return img

        image = ensure_shape(image)
        yolo_image = ensure_shape(yolo_image)
        heatmap = ensure_shape(heatmap)

        image, yolo_image, power_image, decider_img = self.get_decision(image, yolo_image,power_detection_img, yolo_rect_conf, power_rect, heatmap)

        power_image = ensure_shape(power_image)
        yolo_image = cv2.flip(yolo_image, 1)
        combined = image
        combined = cv2.addWeighted(combined, 1, yolo_image, 0.7, 0)
        combined = cv2.addWeighted(combined, 1, power_image, 0.7, 0)
        combined = cv2.addWeighted(combined, 1, heatmap, 0.7, 0)
        # combined = cv2.addWeighted(combined, 1, decider_img, 0.7, 0)


        combined_resized = cv2.flip(combined, 1)  # Flip combined image

        return combined_resized
    
    def get_decision(self, image, yolo_image, power_detection_img, yolo_rect_conf, power_rect, heatmap):
        #Start by checking the light level
        light_level = self.get_lightlevel(image)    
        blank = np.zeros_like(image)
        best_rect = [[0, 0], [0, 0], 0]
        power_image = np.array([1])
        print(light_level)
        if light_level < 0.2:
            print("Low light level, using heatmap")
            yolo_image = blank.copy()

        #Now check the heatmap for entropy (multiple sources of sound or not)
        entropy_conf = self.get_entropy(heatmap)
        print("Entropy confidence:", entropy_conf)
        if entropy_conf < 0.076:
            power_image = blank.copy()
            #print("Low entropy, using yolo detection")

        #Get yolo rect with highest confidence
        if yolo_rect_conf is None or len(yolo_rect_conf) == 0:
            print("No YOLO detection, using power rect")
            yolo_rect_conf = [[0, 0], [0, 0], 0]
        elif yolo_image.any():
            best_rect = max(yolo_rect_conf, key=lambda x: x[2]) 
            yolo_image = self.create_rect(blank.copy(), (best_rect[0][0], best_rect[0][1], best_rect[1][0], best_rect[1][1]))
        #Get Intersection over union (how much the boxes overlap)
        iou = self.get_iou((best_rect[0][0], best_rect[0][1], best_rect[1][0], best_rect[1][1]), power_rect)
        if iou > 0.9:
            #High iou, using the center between rectangles
            x1, y1, x2, y2 = (yolo_rect_conf[0][0], yolo_rect_conf[0][1], yolo_rect_conf[1][0], yolo_rect_conf[1][1])
            x3, y3, x4, y4 = power_rect
            x_mid1 = (x1 + x2) / 2
            y_mid1 = (y1 + y2) / 2
            x_mid2 = (x3 + x4) / 2
            y_mid2 = (y3 + y4) / 2
            decider_img = cv2.rectangle(blank.copy(), (int(x_mid1), int(y_mid1)), (int(x_mid2), int(y_mid2)), (100, 255, 255), 2)
        else : decider_img = blank
        if power_image.any():
            power_image = self.create_rect(power_detection_img, (power_rect[0], power_rect[1], power_rect[2], power_rect[3]), color=(0, 255, 100))


        
        return image, yolo_image, power_image, decider_img
    
    def focus_beam(self, callback, box):
        x1, y1, x2, y2, conf = box
        if conf < self.image_confidence_threshold:
            print("Low confidence, not focusing beam")
            return -1, -1
        else:
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            horizontal = (x_mid / self.display_size[0]) * self.MAX_X * 2 - self.MAX_X
            vertical = (y_mid / self.display_size[1]) * self.MAX_Y * 2 - self.MAX_Y
            callback(horizontal, vertical)



            # # steer(-horizontal, vertical)
            # print(f"{horizontal}, {vertical}")
            # self.cb(horizontal, vertical)

        return 0
    
    def get_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # Calculate intersection
        inter_x1 = max(x1, x3)
        inter_y1 = max(y1, y3)
        inter_x2 = min(x2, x4)
        inter_y2 = min(y2, y4)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (x4 - x3) * (y4 - y3)
            iou = intersection_area / float(box1_area + box2_area - intersection_area)
            return iou
        return 0.0

    def create_rect(self, image, rect, color=(255, 0, 255)):
        if len(rect) == 0:
            return image
        x1, y1, x2, y2 = rect
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        return image


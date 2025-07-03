import cv2
import numpy as np
from time import time

class sensorfusiondecider:
    def __init__(self, display_size=(640, 360), MAX_ANGLE=30, ASPECT_RATIO=16/9):
        self.display_size = display_size
        self.image_confidence_threshold = 0.5
        self.MAX_X = MAX_ANGLE
        self.MAX_Y = MAX_ANGLE / ASPECT_RATIO
        self.last_known_position = None
        self.last_known_time = None
        self.max_pixels_per_second = 1000


        
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
    
    def ensure_shape(self, img):
        img = cv2.resize(img, self.display_size)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.dtype != np.uint8:
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8) if img.dtype == np.float32 else img.astype(np.uint8)
        return img
    
    def flip_rect_horizontally(self, rect, image_width):
        x1, y1, x2, y2 = rect
        return (image_width - x2, y1, image_width - x1, y2)
    
    def create_image(self, image, yolo_image,power_detection_img, yolo_rect_conf, power_rect, heatmap):

        # Ensure all images are 3-channel and same size

        image = self.ensure_shape(image)
        yolo_image = self.ensure_shape(yolo_image)
        heatmap = self.ensure_shape(heatmap)

        # image, yolo_image, power_image, decider_img = self.get_decision(image, yolo_image,power_detection_img, yolo_rect_conf, power_rect, heatmap)

        # power_image = ensure_shape(power_image)
        #yolo_image = cv2.flip(yolo_image, 1)
        combined = image
        power_rect = sensorfusiondecider.rescale_rect(
            power_rect,
            power_detection_img.shape[:2],
            image.shape[:2]
        )
        for i, rect in enumerate(yolo_rect_conf):
            rect_flipped = self.flip_rect_horizontally(rect[:4], image.shape[1])
            yolo_rect_conf[i] = [rect_flipped[0], rect_flipped[1], rect_flipped[2], rect_flipped[3], rect[4]]
        # combined = cv2.addWeighted(combined, 1, yolo_image, 0.7, 0)
        # combined = cv2.addWeighted(combined, 1, power_image, 0.7, 0)
        # combined = cv2.addWeighted(combined, 1, heatmap, 0.7, 0)
        # combined = cv2.addWeighted(combined, 1, decider_img, 0.7, 0)
        print("yolo rect confidence:", yolo_rect_conf)
        decider_img = self.get_decision(image, yolo_image, power_detection_img, yolo_rect_conf, power_rect, heatmap)
        combined = decider_img

        combined_resized = cv2.flip(combined, 1)  # Flip combined image

        return combined_resized
    
    def get_decision_(self, image, yolo_image, power_detection_img, yolo_rect_conf, power_rect, heatmap):
        #Start by checking the light level
        light_level = self.get_lightlevel(image)    
        blank = np.zeros_like(image)
        best_rect = [[0, 0], [0, 0], 0]
        yolo_image_use = True
        power_image_use = True
        power_image = np.array([1])
        print(light_level)
        if light_level < 0.2:
            print("Low light level, using heatmap")
            yolo_image_use = False

        #Now check the heatmap for entropy (multiple sources of sound or not)
        entropy_conf = self.get_entropy(heatmap)
        print("Entropy confidence:", entropy_conf)
        if entropy_conf < 0.076  :
            power_image_use = False
            #print("Low entropy, using yolo detection")

        #Get yolo rect with highest confidence
        if yolo_rect_conf is None or len(yolo_rect_conf) == 0:
            print("No YOLO detection, using power rect")
            yolo_rect_conf = [[0, 0], [0, 0], 0]
        elif yolo_image_use:
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
    
    def get_decision(self, image, yolo_image, power_detection_img, yolo_rect_conf, power_rect, heatmap):
        yolo_image_use = True
        power_image_use = True
        best_rect = [0, 0, 0, 0, 0, 0]
        blank = np.zeros_like(image)
        decider_img = image.copy()
        yolo_rect_conf = np.array(yolo_rect_conf)
        iou = 0.0

        #Start by checking the light level
        light_level = self.get_lightlevel(image)
        print("Light level:", light_level)
        if light_level < 0.2 or not yolo_rect_conf.all():
            print("Low light level or no yolo rects, using heatmap")
            yolo_image_use = False
        
        #Now check the heatmap for entropy (multiple sources of sound or not)
        entropy_conf = self.get_entropy(heatmap)
        print("Entropy confidence:", entropy_conf)
        if entropy_conf < 0.076:
            power_image_use = False
            print("High entropy, using yolo detection")
        
        #if both false, decider image will be image
        if not yolo_image_use and not power_image_use:
            print("No valid detection, using image")
            decider_img = image.copy()
            return decider_img
        
        #if yolo is true, check if rect has done a possible movement

        #if yolo true power false, use all yolo rects
        if yolo_image_use and not power_image_use:
            print("Using yolo detection only")
            for rect in yolo_rect_conf:
                x1, y1, x2, y2, conf = rect
                if conf < self.image_confidence_threshold:
                    continue
                decider_img = self.create_rect(decider_img, (x1, y1, x2, y2), color=(255, 0, 255))
            return decider_img


        #if power true yolo false, use power rect
        elif not yolo_image_use and power_image_use:
            print("Using power detection only")
            x1, y1, x2, y2 = power_rect
            return self.create_rect(decider_img, (x1, y1, x2, y2), color=(0, 0, 255))
        
        #Check how many yolos
        amount_inferences = len(yolo_rect_conf)

        #If one and both use = true, check iou
        if amount_inferences == 1 and yolo_image_use and power_image_use:
            print("One yolo detection, checking iou")
            iou = self.get_iou(yolo_rect_conf[0], power_rect)

        #if no iou and still one, use yolo
        if iou < 0.1 and amount_inferences == 1:
            print("No iou, using yolo detection")
            x1, y1, x2, y2, conf = yolo_rect_conf[0][:5]
            decider_img = self.create_rect(decider_img, (x1, y1, x2, y2), color=(255, 0, 255))
            return decider_img
        
        #if iou and still one, use intersection
        elif iou > 0.1 and amount_inferences == 1:
            print("High iou, using intersection")
            rect = self.get_intersecting_rect(yolo_rect_conf[0], power_rect)
            x1, y1, x2, y2 = rect
            decider_img = self.create_rect(decider_img, (x1, y1, x2, y2), color=(255, 0, 0))
            return decider_img

        #if no iou and several yolo, get closest yolo weighted by confidence

        elif iou < 0.1 and amount_inferences > 1:
            print("No iou, using closest yolo detection")
            closest_box = self.get_closest_rect(yolo_rect_conf, power_rect)
            x1, y1, x2, y2, conf = closest_box
            decider_img = self.create_rect(decider_img, (x1, y1, x2, y2), color=(255, 0, 255))
            return decider_img


    
    def focus_beam(self, callback, box):
        x1, y1, x2, y2, conf = box[:5]
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
        x1, y1, x2, y2 = box1[:4]
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

    def get_intersecting_rect(self, box1, box2):
        x1, y1, x2, y2 = box1[:4]
        x3, y3, x4, y4 = box2[:4]

        # Calculate center points of both boxes
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        cx2 = (x3 + x4) / 2
        cy2 = (y3 + y4) / 2

        # Create a rectangle between the two center points
        x_min = int(min(cx1, cx2))
        y_min = int(min(cy1, cy2))
        x_max = int(max(cx1, cx2))
        y_max = int(max(cy1, cy2))

        return (x_min, y_min, x_max, y_max)
    
    def get_closest_rect(self, boxes_confidence, target_rect):
        closest_box = None
        min_distance = float('inf')
        if len(boxes_confidence[0]) == 5:
            for box in boxes_confidence:
                x1, y1, x2, y2, conf = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                target_center_x = (target_rect[0] + target_rect[2]) / 2
                target_center_y = (target_rect[1] + target_rect[3]) / 2
                distance = np.sqrt((center_x - target_center_x) ** 2 + (center_y - target_center_y) ** 2) / (conf + 1e-6)

                if distance < min_distance:
                    min_distance = distance
                    closest_box = box

            return closest_box if closest_box is not None else [0, 0, 0, 0, 0, 0]
        else:
            for box in boxes_confidence:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                target_center_x = (target_rect[0] + target_rect[2]) / 2
                target_center_y = (target_rect[1] + target_rect[3]) / 2
                distance = np.sqrt((center_x - target_center_x) ** 2 + (center_y - target_center_y) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    closest_box = box

            return closest_box if closest_box is not None else [0, 0, 0, 0]
    
    def rescale_rect(rect, from_shape, to_shape):
        """Rescale a rectangle from one image shape to another."""
        x1, y1, x2, y2 = rect[:4]
        from_w, from_h = from_shape
        to_w, to_h = to_shape
        scale_x = to_w / from_w
        scale_y = to_h / from_h
        return (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        )


    def is_plausible_movement(self, new_rect, old_rect, delta_time):
        """Returns True if the movement is plausible based on speed constraint."""
        if old_rect is None or delta_time is None:
            return True  # First detection always passes
        x1, y1, x2, y2 = new_rect[:4]
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2

        ox1, oy1, ox2, oy2 = old_rect[:4]
        cx2 = (ox1 + ox2) / 2
        cy2 = (oy1 + oy2) / 2

        distance = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        return (distance / delta_time) <= self.max_pixels_per_second




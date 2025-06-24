import cv2

import numpy as np
import matplotlib.pyplot as plt

from interface.config import *

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
WINDOW_DIMENSIONS = (1920,1080)
APPLICATION_NAME = "Demo App"
HEATMAP_COLOR = False
NUM_WINDOWS = 1
POWER = 5

SRC = "/dev/video2" # This was our webcam

SRC = -1 # This will give any webcam

try:
    # Try to import kalman filter for tracking, but no method is using it at the moment
    from lib.kf import *
    kf = CyKF()
except ModuleNotFoundError:
    print("Unable to find kalman filter, has it been compiled?")

def generate_color_map(name="jet") -> np.ndarray:
    """Create a color lookup table for values between 0 - 255

    Args:
        name (str, optional): Matplotlib CMap. Defaults to "jet".

    Returns:
        np.ndarray: the lookup-table
    """
    
    cmap = plt.cm.get_cmap(name)

    # cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()

def local_max(img: np.ndarray, threshold: float) -> np.ndarray:
    padded_img = np.pad(img, ((1,1),(1,1)), constant_values=-np.inf)

    # Determines if each location is bigger than adjacent neighbors
    adjacentmax =(
    (padded_img[1:-1,1:-1] > threshold) &
    (padded_img[0:-2,1:-1] <= padded_img[1:-1,1:-1]) &
    (padded_img[2:,  1:-1] <= padded_img[1:-1,1:-1]) &
    (padded_img[1:-1,0:-2] <= padded_img[1:-1,1:-1]) &
    (padded_img[1:-1,2:  ] <= padded_img[1:-1,1:-1])
    )

    return adjacentmax

def calculate_heatmap2(image, threshold=1e-7, amount = 0.5, exponent = POWER):
    """Create a heatmap over the perceived powerlevel

    Args:
        image (np.ndarray[MAX_RES_X, MAX_RES_Y]): The calculated powerlevels for each anlge
        threshold (float, optional): minimum max value to print out. Defaults to 5e-8.

    Returns:
        (heatmap, bool): the calculated heatmap and if it should be output or not
    """
    # placeholder
    should_overlay = False
    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    # bool_map = np.ones_like(image) * local_max(image, threshold)
    ind = np.unravel_index(np.argmax(image, axis=None), image.shape)

    x, y = ind

    kf.update([x, y, 0])

    x, y, _ = kf.get_state()
    power_level = np.max(image)
    color_val = int(255 * power_level ** exponent)
    if x < 0:
        x = 0
    elif x >= MAX_RES_X:
        x = MAX_RES_X - 1
    else:
        x = int(x)

    if y < 0:
        y = 0
    elif y >= MAX_RES_Y:
        y = MAX_RES_Y - 1
    else:
        y = int(y)
    # small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[color_val]

    
    x1 = x
    y1 = y

    max_power_level = np.max(image)


    # Normalize the image
    image /= max_power_level

    if max_power_level > threshold:

        should_overlay = True
        # Convert image value in range between [0, 1] to a RGB color value
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                power_level = image[x, y]

                # Only paint levels above a certain amount, i.e 50%
                if power_level >= amount:
                    power_level -= amount
                    power_level /= amount

                    # Some heatmaps are very flat, so the power of the power
                    # May give more sharper results
                    color_val = int(255 * power_level ** exponent)

                    # This indexing is a bit strange, but CV2 orders it like this (Same as flip operation)
                    small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[color_val]

        
    # Must resize to fit camera dimensions
    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    X = WINDOW_DIMENSIONS[0] - 1 - int(x1 / (MAX_RES_X - 1) * WINDOW_DIMENSIONS[0])
    Y = WINDOW_DIMENSIONS[1] - 1 - int(y1 / (MAX_RES_Y - 1) * WINDOW_DIMENSIONS[1])
    cv2.circle(heatmap,(X, Y), 50, (0,255,0), 5)
    return heatmap, should_overlay


def calculate_heatmap(image, threshold=1e-7, amount = 0.5, exponent = POWER):
    """Create a heatmap over the perceived powerlevel

    Args:
        image (np.ndarray[MAX_RES_X, MAX_RES_Y]): The calculated powerlevels for each anlge
        threshold (float, optional): minimum max value to print out. Defaults to 5e-8.

    Returns:
        (heatmap, bool): the calculated heatmap and if it should be output or not
    """
    should_overlay = False
    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    
    max_power_level = np.max(image)
    if image.ndim == 3:
        image = image[..., 0]
    safe_image = np.clip(image, 1e-12, None)


    if (max_power_level > threshold):

        img = np.log10(safe_image)
        img -= np.log10(np.min(safe_image))
        img /= np.max(img)

        should_overlay = True
        # Convert image value in range between [0, 1] to a RGB color value
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                power_level = img[x, y]

                # Only paint levels above a certain amount, i.e 50%
                if (power_level >= amount):
                    power_level -= amount
                    power_level /= amount

                    # Some heatmaps are very flat, so the power of the power
                    # May give more sharper results
                    color_val = int(255 * power_level ** exponent)

                    # This indexing is a bit strange, but CV2 orders it like this (Same as flip operation)
                    small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[color_val]
    # Must resize to fit camera dimensions
    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    
    return heatmap, should_overlay

def calculate_heatmap_fft(image, threshold=5e-8):
    """"""
    lmax = np.max(image)

    # print(lmax)

    # image[image < threshold] = 0.0

    image /= lmax
    should_overlay = False

    # image = image.T

    small_heatmap = np.zeros((11, 11, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>threshold*1000000:
        for x in range(11):
            for y in range(11):
                d = image[x, y]

                if d >= 0.5:
                    d -= 0.5
                    d*= 2
                    val = int(255 * d ** 2)

                    small_heatmap[11 - 1 - y, 11 - 1 - x] = colors[val]
                    should_overlay = True

    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap, should_overlay

import numpy as np
import cv2
from interface.config import *
from interface.config import MAX_RES_X, MAX_RES_Y, MAX_ANGLE, ASPECT_RATIO
def calculate_heatmap_with_detection(image, threshold=1e-7, amount=0.5, exponent=POWER, box_size_ratio=0.1, region_size=3):
    """Create a heatmap with bounding box around detected object
    
    Args:
        image: Power level array [MAX_RES_X, MAX_RES_Y]
        box_size_ratio: Size of bounding box as ratio of image size
        region_size: Size of region to consider for center-of-mass calculation
    """
    
    should_overlay = False
    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    
    max_power_level = np.max(image)
    if image.ndim == 3:
        image = image[..., 0]
    safe_image = np.clip(image, 1e-12, None)
    peak_y, peak_x = find_power_center(safe_image, region_size)

    # Find peak location using center of mass of high-power region
    if HEATMAP_COLOR:
        if max_power_level > threshold:
            # Create heatmap (your existing code)
            img = np.log10(safe_image)
            img -= np.log10(np.min(safe_image))
            img /= np.max(img)
            should_overlay = True
            
            # Generate heatmap colors
            for x in range(MAX_RES_X):
                for y in range(MAX_RES_Y):
                    power_level = img[x, y]
                    if power_level >= amount:
                        power_level -= amount
                        power_level /= amount
                        color_val = int(255 * power_level ** exponent)
                        small_heatmap[MAX_RES_Y - 1 - y, MAX_RES_X - 1 - x] = colors[color_val]
    else:
        should_overlay = True
    
    # Resize to window dimensions
    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    ACTUAL_DISPLAY_SIZE = WINDOW_DIMENSIONS 
    if should_overlay:
        
        # Print original coordinates

        # Calculate the scaled coordinates
        scaled_window_x = ACTUAL_DISPLAY_SIZE[0] - 1 - int(peak_x / (MAX_RES_X - 1) * ACTUAL_DISPLAY_SIZE[0])
        scaled_window_y = ACTUAL_DISPLAY_SIZE[1] - 1 - int(peak_y / (MAX_RES_Y - 1) * ACTUAL_DISPLAY_SIZE[1])

        # Print the scaled coordinates
        # Now check the bounding box coordinates based on these scaled values
        box_width = int(ACTUAL_DISPLAY_SIZE[0] * box_size_ratio)
        box_height = int(ACTUAL_DISPLAY_SIZE[1] * box_size_ratio)

        x1 = max(0, scaled_window_x - box_width // 2)
        y1 = max(0, scaled_window_y - box_height // 2)
        x2 = min(ACTUAL_DISPLAY_SIZE[0], scaled_window_x + box_width // 2)
        y2 = min(ACTUAL_DISPLAY_SIZE[1], scaled_window_y + box_height // 2)

        # Draw bounding box for debugging
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Purple box
        cv2.circle(heatmap, (scaled_window_x, scaled_window_y), 5, (0, 0, 255), -1)  # Red center

        
        # Add confidence text
        confidence = max_power_level / threshold if threshold > 0 else 1.0
        cv2.putText(heatmap, f"Conf: {confidence:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    return heatmap, should_overlay

def find_power_center(image, region_size=3):
    """Find center with OpenCV Gaussian smoothing"""
    # Convert to float32 for OpenCV processing
    image_f32 = image.astype(np.float32)
    
    # Apply Gaussian blur (kernel size should be odd)
    kernel_size = 5  # Adjust for more/less smoothing
    smoothed = cv2.GaussianBlur(image_f32, (kernel_size, kernel_size), sigmaX=1.0, sigmaY=1.0)
    
    max_val = np.max(smoothed)
    threshold = max_val * 0.95
    high_power_mask = smoothed >= threshold
    
    if np.sum(high_power_mask) > 0:
        y_indices, x_indices = np.indices(smoothed.shape)
        
        # Use cubed power for strong weighting
        weights = (smoothed ** 3) * high_power_mask
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            center_x = np.sum(x_indices * weights) / total_weight
            center_y = np.sum(y_indices * weights) / total_weight
            return center_x, center_y
    
    # Fallback
    peak_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    return peak_idx[1], peak_idx[0]

import queue
from multiprocessing import JoinableQueue, Value

class Front:
    def __init__(self, q_rec: JoinableQueue, q_out: JoinableQueue, running: Value, src=SRC):
        self.q_rec = q_rec
        self.q_out = q_out
        self.running = running

        # Setup camera
        self.src = src
        self.capture = cv2.VideoCapture(self.src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def multi_loop(self, *args, **kwargs):
        prev = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.MAX_X = MAX_ANGLE
        self.MAX_Y = MAX_ANGLE /  ASPECT_RATIO
        while self.running:
            try:
                output = self.q_rec.get(block=False)
                self.q_rec.task_done()
                status, frame = self.capture.read()
                frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
                try:
                    frame = cv2.resize(frame, WINDOW_DIMENSIONS)
                except cv2.error as e:
                    print("An error ocurred with image processing! Check if camera and antenna connected properly")
                    self.running.value = 0
                    break

                res1, should_overlay = calculate_heatmap(output, threshold=0)
                res = cv2.addWeighted(prev, 0.5, res1, 0.5, 0)
                prev = res

                if should_overlay:
                    image = cv2.addWeighted(frame, 0.9, res, 0.9, 0)
                else:
                    image = frame
                
                cv2.imshow(APPLICATION_NAME, image)
                cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                self.running.value = 0
                break

    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:

            horizontal = (x / WINDOW_DIMENSIONS[0]) 
            vertical = (y / WINDOW_DIMENSIONS[1]) 
            
            # self.q_out.put((horizontal, vertical))

            # We need to invert Y-axis for the incoming frame since CV2 indexes it as Y - y
            self.q_out.put((vertical, 1.0 - horizontal))
            print(f"{horizontal}, {vertical}")


class Viewer:
    """Test viewer used for outputting calculated heatmaps onto a screen
    """
    def __init__(self, src=SRC, cb=None):
        """constructor with the camera source to use

        Args:
            src (str, optional): which camera index to use. Defaults to "/dev/video2".
        """
        self.src = src
        self.cb = cb
        #self.capture = cv2.VideoCapture(self.src)
        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def loop(self, q_power: JoinableQueue, v: Value, q_viewer: JoinableQueue = None, q_inference: JoinableQueue = None):
        """Threaded or Multiprocessing loop that should not be called by the user

        Args:
            q (JoinableQueue): FIFO containing the latest powermaps from the algorithm
            v (Value): a value that will stop this thread or process when other than 1
            q2 (JoinableQueue, optional): FIFO containing YOLO processed frames
        """
        from sensorfusion.decider import sensorfusiondecider

        decider = sensorfusiondecider()
        prev = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.MAX_X = MAX_ANGLE
        self.MAX_Y = MAX_ANGLE / ASPECT_RATIO
        while v.value == 1:
            try:
                output = q_power.get(block=False)
                yolo_frame_num, yolo_frame = q_inference.get(timeout=0.2)
                q_inference.task_done()
                q_power.task_done()
                viewer_frame_num, frame = q_viewer.get(block=False) if q_viewer is not None else None
                while not (viewer_frame_num == yolo_frame_num):
                    if viewer_frame_num < yolo_frame_num:
                        # Viewer frame is older, skip it
                        try:
                            viewer_frame_num, frame = q_viewer.get(timeout=0.2)
                            q_viewer.task_done()
                        except queue.Empty:
                            break
                    else:
                        # YOLO frame is older, skip it
                        try:
                            yolo_frame_num, yolo_frame = q_inference.get(timeout=0.2)
                            q_inference.task_done()
                            
                        except queue.Empty:
                            break
                if frame is None:
                    frame = np.zeros((APPLICATION_WINDOW_HEIGHT, APPLICATION_WINDOW_WIDTH, 3), dtype=np.uint8)
                frame = cv2.flip(frame, 1) # Nobody likes looking out of the array :(
                try:
                    frame = cv2.resize(frame, WINDOW_DIMENSIONS)
                except cv2.error as e:
                    print("An error ocurred with image processing! Check if camera and antenna connected properly")
                    v.value = 0
                    break

                res1, should_overlay = calculate_heatmap_with_detection(output)

                res = cv2.addWeighted(prev, 0.5, res1, 0.5, 0)
                prev = res
                if HEATMAP_COLOR == False:
                    powerlevel_box = res1
                if should_overlay:
                    image = cv2.addWeighted(frame, 0.9, res, 0.9, 0)
                else:
                    image = frame

                # YOLO inference:
                yolo_image = np.zeros((APPLICATION_WINDOW_HEIGHT, APPLICATION_WINDOW_WIDTH, 3), dtype=np.uint8)
                if q_inference is not None:
                    try:
                        if yolo_frame is not None:
                            yolo_image = cv2.resize(yolo_frame, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
                    except queue.Empty:
                        print("No YOLO frame received")
                        pass
                
                if NUM_WINDOWS == 2:
                    combined = np.hstack((image, yolo_image))
                    display_size = (1280, 360)  # width, height for the window
                    combined_resized = cv2.resize(combined, display_size)
                    cv2.imshow(APPLICATION_NAME, combined_resized)
                elif NUM_WINDOWS == 1:
                    combined_resized = decider.create_image(image, yolo_image, powerlevel_box)
                    cv2.imshow(APPLICATION_NAME, combined_resized)

                cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                v.value = 0
                break
    def mouse_click_handler(self, event, x, y, flags, params):
        """Steers the antenna to listen in a specific direction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            horizontal = (x / WINDOW_DIMENSIONS[0]) * self.MAX_X * 2 - self.MAX_X
            vertical = (y / WINDOW_DIMENSIONS[1]) * self.MAX_Y * 2 - self.MAX_Y
            # steer(-horizontal, vertical)
            print(f"{horizontal}, {vertical}")
            self.cb(horizontal, vertical)
            print("Steering done")


if __name__ == "__main__":
    class d:
        value = 1

    class q:
        def __init__(self):
            pass

        def get(self, block=False):
            return np.ones((MAX_RES_X, MAX_RES_Y))
        def task_done(self):
            pass

    v = Viewer()

    v.loop(q(), d)
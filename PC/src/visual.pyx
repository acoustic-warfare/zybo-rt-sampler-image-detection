# cython: language_level=3
# distutils: language=c

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
sys.path.insert(0, "") # Access local modules located in . Enables 'from . import MODULE'


from config cimport *

import cv2

WINDOW_DIMENSIONS = (APPLICATION_WINDOW_WIDTH, APPLICATION_WINDOW_HEIGHT)
APPLICATION_NAME = "Demo App"

import matplotlib.pyplot as plt

def generate_color_map(name="jet"):
    
    cmap = plt.cm.get_cmap(name)

    # cdef np.ndarray[np.uint8_t, ndim=2, mode="c"] colors 

    # Generate color lookup table
    colors = np.empty((256, 3), dtype=np.uint8)

    for i in range(256):
        colors[i] = (np.array(cmap(255 - i)[:3]) * 255).astype(np.uint8)

    return colors


colors = generate_color_map()


def calculate_heatmap_old(image):
    """"""
    lmax = np.max(image)

    image /= lmax

    # image = image.T

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[x, y] = colors[val]

    # cv2.imshow()

    # small_heatmap = np.reshape(small_heatmap, (MAX_RES_Y, MAX_RES_X, 3))


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap

def calculate_heatmap(image):
    """"""
    lmax = np.max(image)

    image /= lmax

    # image = image.T

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    # small_heatmap[y, MAX_RES_X - 1 - x] = colors[val]
                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[x, y] = colors[val]

    # cv2.imshow()

    # small_heatmap = np.reshape(small_heatmap, (MAX_RES_Y, MAX_RES_X, 3))


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    return heatmap

def calculate_heatmap2_(img):
    """"""
    lmax = np.max(img)

    # image /= lmax

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    kernel = np.ones((6,6))

    img2 = np.ones_like(img)
    loc_max = cv2.dilate(img, kernel) == img
    res = np.int8(img2 * loc_max)

    if lmax>1e-8:
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = res[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]


    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    

    return heatmap


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


from scipy.ndimage import gaussian_filter
def calculate_heatmap2(img):

    img = gaussian_filter(img, sigma=8)
    peaks = local_max(img, threshold=-np.inf)

    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)

    dd = np.copy(img)

    old_max = dd.max()

    dd /= old_max

    dd **=10

    r = 4

    rang = (np.log10(old_max) + 10) / r

    dd *= rang

    for x in range(MAX_RES_X):
        for y in range(MAX_RES_Y):
            d = dd[x, y]

            if d > 0.4:
                val = max(min(int(255 * d), 255), 0)
                small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]

    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)

    return heatmap

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

def calculate_heatmap_with_detection(image, box_size_ratio=0.1):
    """Create a heatmap with bounding box around detected object using calculate_heatmap as basis"""
    
    peak_x, peak_y = find_power_center(image)
    # Debug: Also find the raw peak for comparison
    lmax = np.max(image)
    
    image /= lmax


    small_heatmap = np.zeros((MAX_RES_Y, MAX_RES_X, 3), dtype=np.uint8)
    # small_heatmap = np.zeros((MAX_RES_X, MAX_RES_Y, 3), dtype=np.uint8)

    should_overlay = False
    if lmax > 1e-8:
        should_overlay = True
        for x in range(MAX_RES_X):
            for y in range(MAX_RES_Y):
                d = image[x, y]

                if d > 0.9:
                    val = int(255 * d ** MISO_POWER)

                    # small_heatmap[y, MAX_RES_X - 1 - x] = colors[val]
                    small_heatmap[MAX_RES_Y - 1 - y, x] = colors[val]
                    # small_heatmap[x, y] = colors[val]

    # cv2.imshow()

    # small_heatmap = np.reshape(small_heatmap, (MAX_RES_Y, MAX_RES_X, 3))

    heatmap = cv2.resize(small_heatmap, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
    # heatmap = cv2.resize(small_heatmap, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    
    ACTUAL_DISPLAY_SIZE = (1280, 360)
    # Draw bounding box around detected object
    if should_overlay:
    
        
        window_x = ACTUAL_DISPLAY_SIZE[0] - 1 - int(peak_x / (MAX_RES_X - 1) * ACTUAL_DISPLAY_SIZE[0])
        window_y = ACTUAL_DISPLAY_SIZE[1] - 1 - int(peak_y / (MAX_RES_Y - 1) * ACTUAL_DISPLAY_SIZE[1])
        
        # Calculate box size based on ACTUAL display
        box_width = int(ACTUAL_DISPLAY_SIZE[0] * box_size_ratio)
        box_height = int(ACTUAL_DISPLAY_SIZE[1] * box_size_ratio)
        
        # Calculate box corners
        x1 = max(0, window_x - box_width // 2)
        y1 = max(0, window_y - box_height // 2)
        x2 = min(ACTUAL_DISPLAY_SIZE[0], window_x + box_width // 2)
        y2 = min(ACTUAL_DISPLAY_SIZE[1], window_y + box_height // 2)
        
        # Draw bounding box BEFORE the final resize
        cv2.rectangle(heatmap, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Purple box
        cv2.circle(heatmap, (window_x, window_y), 5, (0, 0, 255), -1)  # Red center
        print(f"Peak at: ({peak_x}, {peak_y}) -> ({window_x}, {window_y})")

        
        # Add confidence text
        confidence = max_power_level / threshold if threshold > 0 else 1.0
        cv2.putText(heatmap, f"Conf: {confidence:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    return heatmap, should_overlay
import queue
from multiprocessing import JoinableQueue, Value


class Viewer:
    def __init__(self, src="/dev/video2"):
        self.src = src
        #self.capture = cv2.VideoCapture(self.src)
        #self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, APPLICATION_WINDOW_WIDTH)
        #self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, APPLICATION_WINDOW_HEIGHT)
        #self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def loop(self, q_power: JoinableQueue, v: Value, q_viewer: JoinableQueue = None, q_inference: JoinableQueue = None):
        """Threaded or Multiprocessing loop that should not be called by the user

        Args:
            q (JoinableQueue): FIFO containing the latest powermaps from the algorithm
            v (Value): a value that will stop this thread or process when other than 1
            q2 (JoinableQueue, optional): FIFO containing YOLO processed frames
        """
        prev = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.MAX_X = MAX_ANGLE
        self.MAX_Y = MAX_ANGLE / ASPECT_RATIO
        while v.value == 1:
            try:
                output = q_power.get(block=False)
                q_power.task_done()
                frame = q_viewer.get(block=False) if q_viewer is not None else None
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

                if should_overlay:
                    image = cv2.addWeighted(frame, 0.9, res, 0.9, 0)
                else:
                    image = frame

                # YOLO inference:
                yolo_image = np.zeros((APPLICATION_WINDOW_HEIGHT, APPLICATION_WINDOW_WIDTH, 3), dtype=np.uint8)
                if q_inference is not None:
                    try:
                        yolo_frame = q_inference.get(timeout=0.2)
                        q_inference.task_done()
                        if yolo_frame is not None:
                            yolo_image = cv2.resize(yolo_frame, WINDOW_DIMENSIONS, interpolation=cv2.INTER_LINEAR)
                    except queue.Empty:
                        print("No YOLO frame received")
                        pass
                

                combined = np.hstack((image, yolo_image))
                display_size = (1280, 360)  # width, height for the window
                combined_resized = cv2.resize(combined, display_size)
                cv2.imshow(APPLICATION_NAME, combined_resized)

                cv2.setMouseCallback(APPLICATION_NAME, self.mouse_click_handler)
                cv2.waitKey(1)
            except queue.Empty:
                pass
            except KeyboardInterrupt:
                v.value = 0
                break
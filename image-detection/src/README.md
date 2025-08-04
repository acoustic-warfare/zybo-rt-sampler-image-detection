
# yolo_smooth_tracking.py

## Overview

This file provides a set of functions for drone detection and tracking in video streams using the YOLO object detection model and the SORT tracking algorithm. It is designed to process video frames, detect drones, and track their movement over time, with options for recording output and displaying results.

There are some comments in the code it self but here is a readme that explains it a little bit further.'

## Main Components

### 1. `yolo_model` Class
- Wraps the YOLO model for easy loading and inference.
- `get_detections(frame, conf_threshold)`: Runs YOLO on a frame and returns bounding boxes with confidence scores above the threshold.

### 2. Helper Functions
- `compute_iou(box1, box2)`: Calculates Intersection over Union (IoU) between two bounding boxes.
- `extract_patch(frame, box, scale)`: Extracts a region from a frame around a bounding box, optionally scaled.
- `cross_correlation_score(prev_patch, curr_patch)`: Computes similarity between two image patches.
- `track_with_correlation(prev_frame, curr_frame, prev_box)`: Attempts to track an object by matching its appearance between frames.

### 3. Main Processing Functions
- `process_video(video_path, model_path, rec=True)`: 
  - Runs YOLO detection on each frame.
  - Uses correlation-based tracking when detections are uncertain.
  - Draws bounding boxes and confidence scores.
  - Optionally records output video and displays results.

- `process_video_track(video_path, model_path, rec=True)`:
  - Uses YOLO for detection and SORT for tracking.
  - Draws tracked bounding boxes with unique IDs and confidence scores.
  - Optionally records output video and displays results.
  - Falls back to correlation-based tracking if detections are weak.

- `process_video_track_boxes_only(frame_queue, output_queue, ...)`:
  - Designed for multiprocessing scenarios.
  - Processes frames from a queue, applies YOLO and SORT, and outputs tracked boxes and their coordinates.
  - Optionally displays results.

## Usage

- The function process_video_track_boxes_only is used in correlation with the acoustic warfare to detect drones both via sound and video. 

## Notes
- The funtions process_video and process_video_track are dev functions and not used in the program
- The file supports both direct detection and tracking, as well as fallback tracking using image correlation.
- Output videos are saved as `output3.mp4` or `output4.mp4` depending on the function.
- The code is modular and can be adapted for real-time or batch processing.


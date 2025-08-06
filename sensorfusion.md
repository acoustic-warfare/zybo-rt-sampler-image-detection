# Project Report: Sensor Fusion for Drone Detection

## Introduction
The purpose of this project was to create a drone detection software by continuing work on the project acoustic warfare, by using a camera to fuse image detection and beamforming into a system that works bether than the sum of its parts. 
## System Architecture
Acoustic warfare makes use of 3x microphone arrays consisting of 64 microphones each, as well as an fpga (field programmable gate array) while the image detection makes use of a webcamera fastened in the middle of the middlemost microphone array. All of this is then run on a computer with an RTX 500 gpu.

## YOLO model 
The YOLO model is trained on a labeled dataset that contains the following:
Training set : 5367 images
Validation set : 706 images
Test set : 759 images 

Every image has a corresponding txt-file that is the label. They look like this:

0 0.4015625 0.8006944444444445 0.133203125 0.15694444444444444

The first number corresponds to the Category and the rest is the cordinates of the bounding box of the object. YOLO-standard.

The model that is included in the system is a YOLO11n. Where n stands for nano. It is trained for 50 epochs and had a batchsize of 32 and a imgsz of 320. More information on the trained model is avalible in ;"image-detection/model/yolov11trainedmodel". There can you also find the result of the training of that model. ("image-detection/src/yolov11trainedmodel/results.png"). There is a README-file for the yolo_smooth_tracking-file that descibes that file better and it is located here ("image-detection/src/README.md").




## Methodology
The data used in the sensorfusion is gathered by the microphone arrays and camera, where acoustic warfare takes the sound taken by the arays and performs beamforming to find DOA(direction of arrival). This is then output into a heatmap for where the drone is likeliest to be. The image detection gets frames from the camera which yolo the processes and performs image detection on. In order to be able to fuse the readings from both sensors, a bounding box is created around the power peak of the heatmap. 

By analysing the heatmap we're able to find out the entropy of the incoming sound, thereby determining whether there is a single sound source, several, or if there is sound interference that will lead to decreased ability to find the drone through sound. We also analyse the lifght level the camera is capturing, as well as yolo giving a confidence score in the found drone inference. 

## Implementation
Using the variables we receive from our sensors, we can determine whether using image detection or the heatmap is the better choice at the moment. This means that when there are several drones flying, we use image detection. If it is dark, we use sound. Finally, we use a kalman filter to process both the heatmap rectangle and the rectangle we receive from yolo in order to be able to predict where the drone will be as well as making sure the interference from sound bouncing(leading to slightly erratic heatmap, thus slightly erratic heatmap rectangle) is lessened, making the tracking much smoother.



- Software tools and libraries
- Integration of sensors
- Key challenges and solutions

## Results
- Sample outputs (images, sound waveforms, detection events)

## Conclusion
Summary of findings, limitations, and future work.

## References
List of relevant papers, documentation, and resources.
# Image Detection Project

## Overview

This project performs image detection and combines it with acoustic warfare in a sensorfusion. Using the camera used by acoustic warfare to simultaneously do image detection and then depending on environmental conditions(light level, amount of sound sources) decides which system to use. 

## Prerequisites

- Complete steps in root README
- Required Python packages (see `requirements.txt`)


## Installation

1. Complete steps in root README

2. Install dependencies:
    ```bash
    cd image-detection
    pip install -r requirements.txt
    ```

## Usage

To start the program, run:
```bash
cd zybo-rt-sampler-image-detection/PC
make clean
make
python3 demo.py mimo
```

## Configuration

- In PC/src/main.pyx in def mimo() change REPLAY_MODE variable to false for realtime as well as change source to a dev/video* depending on camerasource number. For recordings, change REPLAY_MODE to true and source to a video file, as well as video_source to a .pcap file. 

## Troubleshooting

- Ensure all dependencies are installed.
- Check hardware connections if using Zybo RT Sampler.

## License

This project is licensed under the MIT License.

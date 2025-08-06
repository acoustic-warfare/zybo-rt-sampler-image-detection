## Prerequisites (Ubuntu 22.04)
    *Virtual environment
    * pip install -r ../image-detection/requirements.txt
    * root README



## Building the Program
    Start by reading the README in root. Build and import everything necessary for the foundational system acoustic-warfare (zybo-rt-sampler). 


### Run the Program
make clean + make + python3 demo.py mimo will run the program. In main.pyx mimo() you can change video source, whether to use replay mode or not, and if replay mode is true, what pcap source to use.

## Known Issues
The camera used for the project had a fov of 63, while the beamforming had an fov of 45, meaning the heatmap generated becomes offset by a little bit at the edges.
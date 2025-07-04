# cython: language_level=3
# distutils: language=c

__doc__ = """
This is the Python -> Cython -> C interface used to communicate with the microphone_array,
perform beamforming, start receivers and playbacks.

The reason for all functionality to be located in this file is because the functions inside
needs to communicate with each other in the C scope. Therefore, they won't share variables
if they are located in different runtimes. As the user, you may simply inside your python
interpreter or python program use:

>>> from beamformer import WHAT_YOU_NEED_TO_IMPORT

Most of the functionality can be further explained in src/api.c which this file "sits" on top
of.
"""

import numpy as np
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the numpy PyArray_* API.
np.import_array()

import sys
# Access local modules located in . Enables 'from . import MODULE'
sys.path.insert(0, "") 

# Create specific data-types "ctypedef" assigns a corresponding compile-time type to DTYPE_t.
ctypedef np.float32_t DTYPE_t

# Constants
DTYPE_arr = np.float32

try:
    from lib.directions import calculate_coefficients, active_microphones, compute_convolve_h, calculate_delay_miso, calculate_delays
except:
    print("You must build the directions library")
    sys.exit(1)

# Import configuration variables from config.pxd <- config.h
from config cimport *

# API must contain all C functions that needs IPC
cdef extern from "api.h":
    int load(bint)
    void get_data(float *signals)
    void stop_receiving()
    void pad_mimo(float *image, int *adaptive_array, int n)
    void lerp_mimo(float *image, int *adaptive_array, int n)
    void convolve_mimo_vectorized(float *image, int *adaptive_array, int n)
    void convolve_mimo_naive(float *image, int *adaptive_array, int n)
    void load_coefficients2(int *whole_sample_delay, int n)
    void mimo_truncated(float *image, int *adaptive_array, int n)
    void miso_steer_listen(float *out, int *adaptive_array, int n, int steer_offset)
    int load_miso()
    void load_pa(int *adaptive_array, int n)
    void stop_miso()
    void steer(int offset)

# Exposing all pad and sum beamforming algorithms in C
cdef extern from "algorithms/pad_and_sum.h":
    void load_coefficients_pad(int *whole_samples, int n)
    void load_coefficients_pad2(int *whole_miso, int n)
    void unload_coefficients_pad()
    void unload_coefficients_pad2()
    void pad_delay(float *signal, float *out, int pos_pad)
    void miso_pad(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_pad(float *signals, float *image, int *adaptive_array, int n)

# Exposing all convolve and sum beamforming algorithms in C
cdef extern from "algorithms/convolve_and_sum.h":
    void convolve_delay_naive_add(float *signal, float *h, float *out)
    void convolve_delay_vectorized(float *signal, float *h, float *out)
    void convolve_delay_vectorized_add(float *signal, float *h, float *out)
    void convolve_delay_naive(float *signal, float *out, float *h)
    void convolve_naive(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_convolve_naive(float *signals, float *image, int *adaptive_array, int n)
    void miso_convolve_vectorized(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_convolve_vectorized(float *signals, float *image, int *adaptive_array, int n)
    void load_coefficients_convolve(float *h, int n)
    void unload_coefficients_convolve()

# Exposing all convolve and sum beamforming algorithms in C
cdef extern from "algorithms/lerp_and_sum.h":
    void lerp_delay(float *signal, float *out, float h, int pad)
    void miso_lerp(float *signals, float *out, int *adaptive_array, int n, int offset)
    void mimo_lerp(float *signals, float *image, int *adaptive_array, int n)
    void load_coefficients_lerp(float *delays, int n)
    void unload_coefficients_lerp()


# ---- BEGIN LIBRARY FUNCTIONS ----

def connect(replay_mode: bool = False, verbose=True) -> None:
    """
    Connect to a Zybo data-stream

    [NOTICE]

    You must remember to disconnect after you are done, to let the internal C
    child process terminate safely.

    Args:
        replay_mode     bool    True for using replay mode everything 
                                else or nothing will result in using real data

    Kwargs:
        verbose         bool    If you want to display terminal output or not

    """
    assert isinstance(replay_mode, bool), "Replay mode must be either True or False"

    if load(replay_mode * 1) == -1:
        print("Wrong FPGA protocol data format received, disconnecting")
        disconnect()

    if verbose:
        print("Receiver process is forked.\nContinue your program!\n")


def disconnect() -> None:
    """
    Disconnect from a stream

    This is done by killing the child receiving process
    remember to call this function before calling 'exit()'
    
    """
    stop_receiving()


def receive(signals) -> None:
    """
    Receive the N_SAMPLES latest samples from the Zybo.

    [NOTICE]
    This function is "slow" in the regard that is checks if the `signals` is
    of correct data-type and shape, but fine if you only need the latest sample.
 
    It is important to have the correct datatype and shape as defined 
    in src/config.json

    Usage:

        >>>data = np.empty((N_MICROPHONES, N_SAMPLES), dtype=np.float32)
        >>>receive(data)

    Args:
        signals     np.ndarray The array to be filled with the 
                    latest microphone data
    
    """
    assert signals.shape == (N_MICROPHONES, N_SAMPLES), "Arrays do not match shape"
    assert signals.dtype == np.float32, "Arrays dtype do not match"

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] sig = np.ascontiguousarray(signals)
    
    get_data(&sig[0, 0])


# ---- BEGIN BEAMFORMING FUNCTIONS ----

from multiprocessing import JoinableQueue, Process, Value


cdef _convolve_coefficients_load(h):
    cdef np.ndarray[float, ndim=4, mode="c"] f32_h = np.ascontiguousarray(h)
    load_coefficients_convolve(&f32_h[0, 0, 0, 0], int(h.size))


cdef void _loop_mimo_pad(q: JoinableQueue, running: Value):
    """Producer loop for MIMO using pad-delay algorithm"""
    power_framenr = 0
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # Setting up output buffer
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] power_map
    _power_map = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    power_map = np.ascontiguousarray(_power_map)

    while running.value:
        try:
            pad_mimo(&power_map[0, 0], &active_micro[0], int(n_active_mics))
            power_framenr += 1
            q.put((power_map, power_framenr))
        except:
            break
    
    # Unload when done
    unload_coefficients_pad()

cdef void _loop_miso_pad(q: JoinableQueue, running: Value):
    """Consumer loop for MISO using pad-delay algorithm"""

    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    n_active_mics = 64
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(n_active_mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing

    while running.value:
        try:
            (x, y) = q.get()
            q.task_done()
            stear_miso_beam(x, y)
        except Exception as e:
            print(e)

    print("Cython: Stopping audio playback")
    stop_miso()
    unload_coefficients_pad()


cdef void _loop_miso_lerp(q: JoinableQueue, running: Value):
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[float, ndim=3, mode="c"] f32_fractional_samples
    fractional_samples = calculate_delays()

    f32_fractional_samples = np.ascontiguousarray(fractional_samples.astype(np.float32))

    load_coefficients_lerp(&f32_fractional_samples[0, 0, 0], fractional_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    n_active_mics = 64
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(n_active_mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing

    while running.value:
        try:
            (x, y) = q.get()
            q.task_done()
            stear_miso_beam(x, y)
        except Exception as e:
            print(e)

    print("Cython: Stopping audio playback")
    stop_miso()
    unload_coefficients_lerp()

cdef void _loop_mimo_and_miso_pad(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples
    whole_samples, fractional_samples = calculate_coefficients()
    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # Setting up output buffer
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] power_map
    _power_map = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    power_map = np.ascontiguousarray(_power_map)

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    mics = n_active_mics
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing


    import queue
    while running.value:
        try:
            pad_mimo(&power_map[0, 0], &active_micro[0], int(n_active_mics))
            q_out.put(power_map)

            try:
                (x, y) = q_steer.get(block=False)
                q_steer.task_done()
                stear_miso_beam(x, y)
            except queue.Empty:
                pass
            except Exception as e:
                print(e)
        except:
            break
    
    # Unload when done
    stop_miso()
    unload_coefficients_pad()

cdef void _loop_mimo_and_miso_lerp(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    # Calculating time delay for each microphone and each direction
    cdef np.ndarray[float, ndim=3, mode="c"] f32_whole_samples
    # whole_samples, fractional_samples = calculate_coefficients()
    whole_samples = calculate_delays()
    f32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.float32))

    # Pass int pointer to C function
    load_coefficients_lerp(&f32_whole_samples[0, 0, 0], whole_samples.size)

    # Finding which microphones to use
    cdef np.ndarray[int, ndim=1, mode="c"] active_micro
    active_mics, n_active_mics = active_microphones()
    active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    # Setting up output buffer
    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] power_map
    _power_map = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    power_map = np.ascontiguousarray(_power_map)

    print("Cython: Starting miso")
    # Setup audio playback (Order is important)
    load_miso()
    mics = 128 # n_active_mics
    print("Cython: enabling microphones")
    load_pa(&active_micro[0], int(mics))

    print("Cython: Steering beam")
    steer_cartesian_degree(0, 0) # Listen at zero bearing


    import queue
    while running.value:
        try:
            lerp_mimo(&power_map[0, 0], &active_micro[0], int(n_active_mics))
            q_out.put(power_map)

            try:
                (x, y) = q_steer.get(block=False)
                q_steer.task_done()
                stear_miso_beam(x, y)
            except queue.Empty:
                pass
            except Exception as e:
                print(e)
        except:
            break
    
    # Unload when done
    stop_miso()
    unload_coefficients_lerp()


cdef void api(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    while running.value:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)
    
    unload_coefficients_pad()

"""
The following functions are producers of data since the only create their coefficients
during call. They are also meant to be run in a separate Process and to be stopped
by the Variable `running`.

All these functions use a queue to put their 


"""

cdef void api_with_miso(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    x = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    mimo_arr = np.ascontiguousarray(x)

    load_miso()
    load_pa(&active_micro[0], int(n_active_mics))
    steer(0)

    steer_cartesian_degree(0, 0)

    while running.value:
        pad_mimo(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

    stop_miso()
    unload_coefficients_pad()

cdef void just_miso(q: JoinableQueue, running: Value):
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    
    load_miso()
    # n_active_mics = 60
    load_pa(&active_micro[0], int(n_active_mics))
    # steer(0) # This will set the offset to zero, which is quite bad

    steer_cartesian_degree(0, 0)

    import time
    while running.value:
        time.sleep(0.1) # Do nothing during loop

    # When not running anymore, stop the audio playback and free the coefficients
    stop_miso()
    unload_coefficients_pad()

cdef void api_convolve(q: JoinableQueue, running: Value):

    cdef np.ndarray[np.float32_t, ndim=2, mode = 'c'] mimo_arr
    
    image = np.zeros((MAX_RES_X, MAX_RES_Y), dtype=DTYPE_arr)
    mimo_arr = np.ascontiguousarray(image)

    h = compute_convolve_h()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    _convolve_coefficients_load(h)

    while running.value:
        convolve_mimo_vectorized(&mimo_arr[0, 0], &active_micro[0], int(n_active_mics))
        q.put(mimo_arr)

    unload_coefficients_convolve()


def steer_cartesian_degree(azimuth: float, elevation: float):
    """Steer a MISO into a specific direction"""

    assert -90<=azimuth<=90, "Invalid range"
    assert -90<=elevation<=90, "Invalid range"

    azimuth += 90
    azimuth /= 180
    azimuth = int(azimuth * MAX_RES_X)
    elevation += 90
    elevation /= 180
    elevation = int(elevation * MAX_RES_Y)

    _, n_active_mics = active_microphones()

    steer_offset = int(elevation * MAX_RES_X * n_active_mics + azimuth * n_active_mics)
    
    steer(steer_offset)

def stear_miso_beam(azimuth: float, elevation: float):
    """Steer a MISO into a specific direction"""
    
    azimuth = int(azimuth * MAX_RES_X)
    elevation = int(elevation * MAX_RES_Y)

    _, n_active_mics = active_microphones()

    steer_offset = int(elevation * MAX_RES_X * n_active_mics + azimuth * n_active_mics)
    
    print(steer_offset)
    steer(steer_offset)


cdef void api_miso(q: JoinableQueue, running: Value):
    cdef np.ndarray[np.float32_t, ndim=1, mode = 'c'] out = np.ascontiguousarray(np.zeros(N_SAMPLES, dtype=DTYPE_arr))
    
    whole_samples, fractional_samples = calculate_coefficients()
    active_mics, n_active_mics = active_microphones()

    cdef np.ndarray[int, ndim=1, mode="c"] active_micro = np.ascontiguousarray(active_mics.astype(np.int32))

    cdef np.ndarray[int, ndim=3, mode="c"] i32_whole_samples

    i32_whole_samples = np.ascontiguousarray(whole_samples.astype(np.int32))

    # Pass int pointer to C function
    load_coefficients_pad(&i32_whole_samples[0, 0, 0], whole_samples.size)

    while running.value:
        global steer_offset
        miso_steer_listen(&out[0], &active_micro[0], int(n_active_mics), steer_offset)
        q.put(out)



# Web interface
def uti_api(q: JoinableQueue, running: Value):
    api(q, running)

def uti_api_with_miso(q: JoinableQueue, running: Value):
    api_with_miso(q, running)

def conv_api(q: JoinableQueue, running: Value):
    api_convolve(q, running)

def miso_api(q: JoinableQueue, running: Value):
    api_miso(q, running)

def just_miso_api(q: JoinableQueue, running: Value):
    just_miso(q, running)

def b(q: JoinableQueue, running: Value):
    _loop_mimo_pad(q, running)

def just_miso_loop(q: JoinableQueue, running: Value):
    """Dummy loop for testing miso"""
    import time
    while running.value:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            running.value = 0

import os

def get_unique_filename(base_name="output", ext=".mp4", directory="recordings"):
    i = 1
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = f"{base_name}{ext}"
    full_path = os.path.join(directory, filename)
    while os.path.exists(full_path):
        filename = f"{base_name}_{i}{ext}"
        full_path = os.path.join(directory, filename)
        i += 1
    return full_path


# Testing
import cv2
import time
import pyshark


def udp_capture_to_pcap(output_file="udp_capture.pcap", interface="enp0s31f6"):
    import subprocess
    # Save only UDP packets
    cmd = [
        "tshark",
        "-i", interface,
        "-f", "udp",             # Capture filter: only UDP
        "-w", output_file        # Output file
    ]
    print(f"Starting UDP capture on {interface} to {output_file}")
    subprocess.run(cmd)  # This blocks until interrupted

def camera_reader(q_yolo, q_viewer, running, src="/dev/video0"):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Get video FPS
    frame_delay = 1.0 / fps  # Delay between frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #COMMENT OUT IF YOU DON'T WANT TO SAVE VIDEO
    filename = get_unique_filename("output", ".mp4", "recordings")


    #out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 360)) #COMMENT OUT IF YOU DON'T WANT TO SAVE VIDEO
    frame_number = 0
    while running.value:
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            print("No frame captured, resetting video capture")
            continue
        frame = cv2.resize(frame, (640, 360))
        #out.write(frame)#COMMENT OUT IF YOU DON'T WANT TO SAVE VIDEO
        #if not out.isOpened():
            #print("Error: Could not open video writer")
        
        # When producing frames:
        frame_number += 1
        if q_viewer.full():
                try:
                    q_viewer.get_nowait()  # Remove old frame
                except:
                    pass
        q_viewer.put((frame_number, frame))
        if q_yolo.full():
            try:
                q_yolo.get_nowait()  # Remove old frame
            except:
                pass
        q_yolo.put((frame_number, frame))
            
        # Control frame rate
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)
        if elapsed > 10:
            print("Exiting viewer")
            running.value = 0
            break
    print("Exiting camera reader")
    cap.release()
    #out.release()  #COMMENT OUT IF YOU DON'T WANT TO SAVE VIDEO
import subprocess
def start_udpreplay(pcap_file, interface="lo"):
    cmd = ["udpreplay", "-i", interface, pcap_file]
    return subprocess.Popen(cmd)


def mimo():
    from lib.visual import Viewer
    import time
    v = Value('i', 1)
    q_power = JoinableQueue(maxsize=2)
    q_viewer = JoinableQueue(maxsize=2)
    q_yolo = JoinableQueue(maxsize=1)
    q_yolo_inference = None
    #source = "/home/batman/programming/zybo-rt-sampler-image-detection/PC/recordings/vänhög.mp4"
    source = "/dev/video0"  # Use a camera as source, change to your camera device
    pcap_source = "./recordings/vänhögudp_replace.pcap"  # Use a pcap file as source
    cam_proc = Process(target=camera_reader, args=(q_yolo, q_viewer, v, source))
    cam_proc.start()    
    using_yolo = False
    yolo_proc = None
    


    if(True): # Change to False to disable YOLO
        q_yolo_inference = JoinableQueue(maxsize=2)
        import sys
        sys.path.append("../image-detection/src/")
        from yolo_smooth_tracking import process_video_track_boxes_only as process_video
        yolo_proc = Process(target=process_video, args=(q_yolo, q_yolo_inference, True, False, "/home/batman/programming/zybo-rt-sampler-image-detection/image-detection/model/best_of_all.pt"))
        yolo_proc.start()
        using_yolo = True
        

    
    producer = b
    jobs = 1
    viewer = Viewer()
    REPLAY_MODE = False
    connect(replay_mode=REPLAY_MODE)

    try:

        producers = [
            Process(target=producer, args=(q_power, v)),
        ]
        if REPLAY_MODE:
            producers.append(Process(target=start_udpreplay, args=((pcap_source),"lo"), daemon=True))

        # daemon=True is important here
        consumers = [
            Process(target=viewer.loop, args=(q_power , v,q_viewer, q_yolo_inference), daemon=True)
            for _ in range(jobs * 1)
        ]
        filename = get_unique_filename("udp_capture", ".pcap", "recordings")
        #udp_proc = Process(target=udp_capture_to_pcap, args=(filename, "enp0s31f6"), daemon=True) 
        # + order here doesn't matter
        for p in consumers + producers:
            p.start()

        for p in producers:
            p.join()
        if(using_yolo):
            if(yolo_proc.is_alive()):
                yolo_proc.join()
        #udp_proc.start()
        #udp_proc.join()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        v.value = 0
        disconnect()

# ---- Recording functions ----
import csv

def record_webcam(video_path="output.mp4", ts_path="video_timestamps.csv", device=0, duration=10):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    start = time.time()
    frame_number = 0
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "timestamp"])
        while time.time() - start < duration:
            ret, frame = cap.read()
            if np.all(frame[..., 0] == frame[..., 1]) and np.all(frame[..., 1] == frame[..., 2]):
                # skip this frame
                continue
            if not ret:
                break
            timestamp = time.time()
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
            writer.writerow([frame_number, timestamp])
            frame_number += 1
    cap.release()
    out.release()

import pyshark

def record_udp(interface="enp0s31f6", pcap_path="udp_capture.pcap", ts_path="udp_timestamps.csv", duration=10):
    import subprocess
    import threading

    # Start tshark in a subprocess to capture UDP packets
    tshark_cmd = [
        "tshark", "-i", interface, "-f", "udp", "-w", pcap_path
    ]
    tshark_proc = subprocess.Popen(tshark_cmd)
    time.sleep(duration)
    tshark_proc.terminate()

    # After capture, extract timestamps using pyshark
    cap = pyshark.FileCapture(pcap_path, display_filter="udp")
    with open(ts_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["packet_number", "timestamp"])
        for i, pkt in enumerate(cap):
            writer.writerow([i, pkt.sniff_time.timestamp()])
    cap.close()

def record_sensorfusion():
    duration = 60  # seconds
    output_name = get_unique_filename("output", ".mp4", "recordings")
    output_csv = get_unique_filename("video_timestamps", ".csv", "recordings")
    udp_capture_name = get_unique_filename("udp_capture", ".pcap", "recordings")
    udp_csv_name = get_unique_filename("udp_timestamps", ".csv", "recordings")

    p1 = Process(target=record_webcam, args=(output_name,output_csv, 0, duration))
    p2 = Process(target=record_udp, args=("enp0s31f6", udp_capture_name, udp_csv_name, duration))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print("Recording complete.")


# ---- Wrappers for the beamforming loops
def pure_miso_pad(q: JoinableQueue, running: Value):
    _loop_miso_pad(q, running)

def pure_miso_lerp(q: JoinableQueue, running: Value):
    _loop_miso_lerp(q, running)

def multi_pad(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    _loop_mimo_and_miso_pad(q_steer, q_out, running)

def multi_lerp(q_steer: JoinableQueue, q_out: JoinableQueue, running: Value):
    _loop_mimo_and_miso_lerp(q_steer, q_out, running)



def miso():
    producer = multi_lerp
    from lib.visual import Front
    
    # Create some queues for IPC with the new user input directions and
    # Heatmap feed
    q_rec = JoinableQueue(maxsize=2)
    q_out = JoinableQueue(maxsize=2)

    is_running = Value('i', 1)
    f = Front(q_rec, q_out, is_running)
    consumer = f.multi_loop

    print("Cython: Connecting to FPGA")
    connect()

    try:

        producers = [
            Process(target=producer, args=(q_out, q_rec, is_running))

        ]

        # daemon=True is important here
        consumers = [
            Process(target=consumer, daemon=True)
        ]

        # + order here doesn't matter
        for p in consumers + producers:
            p.start()

        for p in producers:
            p.join()


    finally:

        # Stop the program
        is_running.value = 0
        disconnect()

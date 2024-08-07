# Imports
import sys
import pyzed.sl as sl
from signal import signal, SIGINT

# Create camera object
cam = sl.Camera()

# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)

# Set camera parameters
init = sl.InitParameters()
init.depth_mode = sl.DEPTH_MODE.NONE # Set configuration parameters for the ZED

# Open camera
status = cam.open(init) 
if status != sl.ERROR_CODE.SUCCESS: 
    print("Camera Open", status, "Exit program.")
    exit(1)
    
# Set recording parameters
recordingParameters = sl.RecordingParameters()
recordingParameters.compression_mode = sl.SVO_COMPRESSION_MODE.H264
recordingParameters.video_filename = "./data/video_recording.svo"
err = cam.enable_recording(recordingParameters)
if err != sl.ERROR_CODE.SUCCESS:
    print("Recording ZED : ", err)
    exit(1)

# Set runtime parameters
runtime = sl.RuntimeParameters()
frames_recorded = 0
while True:
    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS : # Check that a new image is successfully acquired
        frames_recorded += 1
        print("Frame count: " + str(frames_recorded), end="\r")
# Imports
import sys
import cv2
import numpy as np
from utils import *
import pyzed.sl as sl
from time import sleep
from ultralytics import YOLO
from threading import Lock, Thread

# Set threading params
lock = Lock()
run_signal = False
exit_signal = False

# YOLO detection thread
def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):

	# Set global variables
	global image_net, exit_signal, run_signal, detections

	# Set model
	model = YOLO(weights)

	# Run thread
	while not exit_signal:
		if run_signal:

			# Acquire lock
			lock.acquire()

			# Read image using cv2
			img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
	
			# Predict
			det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes

			# ZED CustomBox format (with inverse letterboxing tf applied)
			detections = detections_to_custom_box(det, image_net)

			# Release lock
			lock.release()

			run_signal = False

		# Sleep
		sleep(0.01)

# Main loop
def main():
	
	# Set global variables
	global image_net, exit_signal, run_signal, detections

	# Set YOLO params
	weights = "./models/train/weights/best.pt"
	img_size = 416
	conf_thres = 0.5

	# Start thread
	capture_thread = Thread(target=torch_thread, kwargs={'weights': weights, 'img_size': img_size, "conf_thres": conf_thres})
	capture_thread.start()

	# Initializing camera
	print("Initializing Camera...")
	zed = sl.Camera()

	# Set input file
	input_type = sl.InputType()
	input_type.set_from_svo_file("./data/test_video_uf_07062024.svo")

	# Set configuration parameters
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
	init_params.coordinate_units = sl.UNIT.METER
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
	init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
	init_params.camera_resolution = sl.RESOLUTION.HD2K # Use HD720 opr HD1200 video mode, depending on camera type.
	init_params.depth_maximum_distance = 50
	init_params.sdk_verbose = 1

	# Set runtime parameters
	runtime_params = sl.RuntimeParameters()
	status = zed.open(init_params)

	# Check camera status
	if status != sl.ERROR_CODE.SUCCESS:
		print(repr(status))
		exit()

	# Camera initialization finished
	print("Initialized Camera")

	# Get camera info
	camera_infos = zed.get_camera_information()
	camera_res = camera_infos.camera_configuration.resolution

	# Get camera calibration params
	calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
	focal_left_x = calibration_params.left_cam.fx # Focal length of the left eye in pixels
	k1 = calibration_params.left_cam.disto[0] # First radial distortion coefficient
	tx = calibration_params.stereo_transform.get_translation().get()[0] # Translation between left and right eye on x-axis
	h_fov = calibration_params.left_cam.h_fov # Horizontal field of view of the left eye in degrees

	# Set positional tracking parameters
	positional_tracking_parameters = sl.PositionalTrackingParameters()
	positional_tracking_parameters.set_as_static = True
	zed.enable_positional_tracking(positional_tracking_parameters)

	# Set object detection parameters
	detection_parameters = sl.ObjectDetectionParameters()
	detection_parameters.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS # choose a detection model
	detection_parameters.enable_tracking = True # Objects will keep the same ID between frames
	detection_parameters.enable_segmentation = False # Outputs 2D masks over detected objects
	zed.enable_object_detection(detection_parameters)

	# Set runtime parameters
	detection_parameters_rt = sl.ObjectDetectionRuntimeParameters()
	detection_parameters_rt.detection_confidence_threshold = 25

	# Utilities for 2D display
	image_left = sl.Mat()
	display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
	image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
	image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

	# Create empty image object and objects object
	image_left_tmp = sl.Mat()
	objects = sl.Objects()

	# Loop
	while not exit_signal:
		if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:

			# Get the image
			lock.acquire()
			zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
			image_net = image_left_tmp.get_data()
			lock.release()
			run_signal = True

			# Detection running on the other thread
			while run_signal:
				sleep(0.001)

			# Wait for YOLO detections
			lock.acquire()
			zed.ingest_custom_box_objects(detections)
			lock.release()

			# Retrieve ZED objects
			zed.retrieve_objects(objects, detection_parameters_rt) 
			zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)

			# 2D rendering
			np.copyto(image_left_ocv, image_left.get_data())
			render_2D(image_left_ocv, image_scale, objects, detection_parameters.enable_tracking)

			# Show image
			cv2.imshow("ZED | 2D View and Birds View", image_left_ocv)
			key = cv2.waitKey(10)
			if key == 27 or key == ord('q') or key == ord('Q'):
				exit_signal = True

		else:
			exit_signal = True

	# Stop loop
	exit_signal = True
	zed.close()

if __name__ == "__main__":
	main()



	
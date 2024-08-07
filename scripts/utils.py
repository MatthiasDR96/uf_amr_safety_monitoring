# Imports
import cv2
import numpy as np
import pyzed.sl as sl



classes = ['AMR', 'Human']
distance_threshold = 1



# Converts yolo xywh to box object for ZED
def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output

# Converts YOLO detections to custom box
def detections_to_custom_box(detections, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def render_object(object_data, is_tracking_on):
    if is_tracking_on:
        return object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK
    else:
        return (object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OK) or (
                    object_data.tracking_state == sl.OBJECT_TRACKING_STATE.OFF)


def draw_vertical_line(left_display, start_pt, end_pt, clr, thickness):
    n_steps = 7
    pt1 = [((n_steps - 1) * start_pt[0] + end_pt[0]) / n_steps
        , ((n_steps - 1) * start_pt[1] + end_pt[1]) / n_steps]
    pt4 = [(start_pt[0] + (n_steps - 1) * end_pt[0]) / n_steps
        , (start_pt[1] + (n_steps - 1) * end_pt[1]) / n_steps]

    cv2.line(left_display, (int(start_pt[0]), int(start_pt[1])), (int(pt1[0]), int(pt1[1])), clr, thickness)
    cv2.line(left_display, (int(pt4[0]), int(pt4[1])), (int(end_pt[0]), int(end_pt[1])), clr, thickness)

def cvt(pt, scale):
    """
    Function that scales point coordinates
    """
    out = [pt[0] * scale[0], pt[1] * scale[1]]
    return out

def get_distance(human, amr):
    return np.round(np.linalg.norm(human.position - amr.position), 2)

def render_2D(left_display, img_scale, objects, is_tracking_on):

    # Init
    overlay = left_display.copy()
    line_thickness = 2

    # Loop over all objects
    for obj in objects.object_list:
        if render_object(obj, is_tracking_on):

            # Get object params
            object_velocity = obj.velocity # Get the object velocity in camera frame
            object_tracking_state = obj.action_state # Get the action state of the object
                    
            # Set color of object based on distance from human to amrs
            if obj.raw_label == 0:

                # Robot color
                base_color = (185, 0, 255) # Blue color

            else:

                # Compute distance from human to all AMRs
                distance = min([get_distance(obj, obj_tmp) for obj_tmp in objects.object_list if obj_tmp.raw_label == 0])

                # Threshold distance
                if distance < distance_threshold:
                    base_color = (232, 176, 59) # Red color
                else:
                    base_color = (175, 208, 25) # Green color

            # Display image scaled 2D bounding box
            top_left_corner = cvt(obj.bounding_box_2d[0], img_scale)
            top_right_corner = cvt(obj.bounding_box_2d[1], img_scale)
            bottom_right_corner = cvt(obj.bounding_box_2d[2], img_scale)
            bottom_left_corner = cvt(obj.bounding_box_2d[3], img_scale)

            # Creation of the 2 horizontal lines
            cv2.line(left_display, (int(top_left_corner[0]), int(top_left_corner[1])), (int(top_right_corner[0]), int(top_right_corner[1])), base_color, line_thickness)
            cv2.line(left_display, (int(bottom_left_corner[0]), int(bottom_left_corner[1])), (int(bottom_right_corner[0]), int(bottom_right_corner[1])), base_color, line_thickness)
            
            # Creation of 2 vertical lines
            draw_vertical_line(left_display, bottom_left_corner, top_left_corner, base_color, line_thickness)
            draw_vertical_line(left_display, bottom_right_corner, top_right_corner, base_color, line_thickness)

            # Scaled ROI
            roi_height = int(top_right_corner[0] - top_left_corner[0])
            roi_width = int(bottom_left_corner[1] - top_left_corner[1])
            overlay_roi = overlay[int(top_left_corner[1]):int(top_left_corner[1] + roi_width), int(top_left_corner[0]):int(top_left_corner[0] + roi_height)]
            overlay_roi[:, :, :] = base_color

            # Text properties
            text_color = base_color
            font_scale = 1
            thickness = 1
            font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            (_, text_height) = cv2.getTextSize("Test", font, font_scale, thickness)[0]

            # Display Object label as text
            text_position = (int(top_left_corner[0]), int(top_left_corner[1]) - 3*text_height)
            text = classes[obj.raw_label] + " - " + str(object_tracking_state)
            cv2.putText(left_display, text, text_position, font, font_scale, text_color, thickness)
            
            # Display Object velocity as text
            text_position = (int(top_left_corner[0]), int(top_left_corner[1]) - 1*text_height)
            text = "Velocity = " + str(np.round(np.linalg.norm(object_velocity), 2)) + " m/s"
            cv2.putText(left_display, text, text_position, font, font_scale, text_color, thickness)

    # Here, overlay is as the left image, but with opaque masks on each detected objects
    cv2.addWeighted(left_display, 0.7, overlay, 0.3, 0.0, left_display)
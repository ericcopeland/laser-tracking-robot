import cv2
import numpy as np


def track_laser(get_cv2_frame_func, process_frame_func, **kwargs):
    '''
    Tracks a laser pointer and returns the frame with the laser pointer circled.
    :param get_cv2_frame_func: A function that returns a cv2 frame (e.g. cv2.imread(...))
    :param process_frame_func: A function that processes a cv2 frame and returns the laser
    pointer min_loc, max_loc, and frame with the laser pointer circled.
    :param kwargs: Additional optional arguments for get_cv2_frame_func and process_frame_func
    :return: The min_loc, max_loc of the laser pointer.
    '''
    frame = get_cv2_frame_func(**kwargs)
    min_loc, max_loc, output = process_frame_func(frame, **kwargs)
    return min_loc, max_loc, output


def gaussian_processing(frame, **kwargs):
    radius = kwargs.get('radius')
    lower_threshold = kwargs.get('lower_threshold')
    upper_threshold = kwargs.get('upper_threshold')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(
        gray_frame,
        (radius, radius),
        0
    )
    _, thresh_inv = cv2.threshold(
        blurred_frame,
        lower_threshold,
        upper_threshold,
        cv2.THRESH_BINARY
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thresh_inv)
    cv2.circle(frame, max_loc, radius, (255, 0, 0), 2)

    return min_loc, max_loc, frame


def hsv_processing(frame, **kwargs):
    low_hsv = np.array(kwargs['low_hsv'])
    high_hsv = np.array(kwargs['high_hsv'])

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_frame, low_hsv, high_hsv)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
    cv2.circle(frame, max_loc, 20, (255, 0, 0), 2)

    return min_loc, max_loc, frame

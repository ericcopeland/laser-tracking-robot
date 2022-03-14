import cv2
import imutils
import numpy as np


def track_object(capture_cv2_frame_func, tracking_func, **options):
    frame = capture_cv2_frame_func(**options)
    return track_object_from_frame(frame, tracking_func, **options)


def track_object_from_frame(frame, tracking_func, prev_frame=None, **tracking_options):
    max_loc, output, processed_output = tracking_func(frame, **tracking_options)
    return max_loc, output, processed_output


def track_objects(capture_cv2_frame_func, tracking_func, **options):
    frame = capture_cv2_frame_func(**options)
    return track_objects_from_frame(frame, tracking_func, **options)


def track_objects_from_frame(frame, tracking_func, prev_frame=None, **tracking_options):
    coords, output, processed_output = tracking_func(frame, prev_frame, **tracking_options)
    return coords, output, processed_output


def gaussian_tracking(frame, prev_frame=None, **tracking_options):
    gaussian_blur_radius = tracking_options.get('gaussian_blur_radius')
    lower_threshold = tracking_options.get('lower_threshold')
    upper_threshold = tracking_options.get('upper_threshold')

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred_frame = cv2.GaussianBlur(
        gray_frame,
        (gaussian_blur_radius, gaussian_blur_radius),
        0
    )
    _, thresh_inv = cv2.threshold(
        blurred_frame,
        lower_threshold,
        upper_threshold,
        cv2.THRESH_BINARY
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(thresh_inv)
    cv2.circle(frame, max_loc, gaussian_blur_radius, (255, 0, 255), 2)

    processed_output = cv2.bitwise_and(gray_frame, thresh_inv)
    processed_output = cv2.cvtColor(processed_output, cv2.COLOR_GRAY2BGR)

    return max_loc, frame, processed_output


def hsv_tracking(frame, prev_frame=None, **tracking_options):
    lower_hsv = tracking_options.get('lower_hsv')
    upper_hsv = tracking_options.get('upper_hsv')

    mask = get_hsv_mask(frame, lower_hsv, upper_hsv)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(mask)
    cv2.circle(frame, max_loc, 2, (255, 0, 0), 2)

    processed_output = cv2.bitwise_and(frame, frame, mask=mask)

    return max_loc, frame, processed_output


def hsv_contour_tracking(frame, prev_frame=None, **tracking_options):
    output = frame.copy() if prev_frame is None else prev_frame

    lower_hsv = tracking_options.get('lower_hsv')
    upper_hsv = tracking_options.get('upper_hsv')

    mask = get_hsv_mask(frame, lower_hsv, upper_hsv)
    processed_output = cv2.bitwise_and(frame, frame, mask=mask)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    cv2.drawContours(
        output,
        contours,
        -1,
        tracking_options.get('outline_color'),
        thickness=tracking_options.get('thickness'),
        lineType=cv2.LINE_AA
    )

    coords = []

    for contour in contours:
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            x = int(moments['m10'] / moments['m00'])
            y = int(moments['m01'] / moments['m00'])
            cv2.circle(output, (x, y), 3, (255, 255, 255), -1)
            coords.append({'left': x, 'top': y})

    return coords, output, processed_output


def get_hsv_mask(frame, lower_hsv, upper_hsv):
    lower_hsv = np.array(lower_hsv)
    upper_hsv = np.array(upper_hsv)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

    return mask

import json
import yaml
import time
import requests

import cv2
import numpy as np

import tracker as tracker
import image_capture as ic

from calibrator import Calibrator
from post_processing import add_post_processing


def parse_options():
    with open('options.yaml', 'r') as stream:
        options = yaml.safe_load(stream)

    options['image_url'] = '{0}://{1}/{2}'.format(
        options['esp32_server']['scheme'],
        options['esp32_server']['host'],
        options['esp32_server']['image_endpoint'],
    )

    options['control_url'] = '{0}://{1}/{2}'.format(
        options['esp32_server']['scheme'],
        options['esp32_server']['host'],
        options['esp32_server']['control_endpoint'],
    )

    orientation = options['frame_stack_orientation']
    options['frame_stack_func'] = np.vstack if orientation == 'vertical' else np.hstack
    options['frame_delay'] = 1 / options['frames_per_second']

    return options


def create_calibrator(capture_type, frame_stack_func, frame_delay, **options):
    capture_options = {}
    capture_cv2_frame_func = None

    if capture_type == 'webcam':
        capture_cv2_frame_func = ic.capture_webcam_cv2_frame_from_stream
        capture_options = {
            'screen_name': 'Webcam Frame Capture',
            'video_capture': cv2.VideoCapture(0)
        }
    elif capture_type == 'esp32':
        capture_cv2_frame_func = ic.capture_esp32_cv2_frame_from_stream
        capture_options = {
            'screen_name': 'ESP32 CAM Frame Capture',
            'image_url': options.get('image_url')
        }
    else:
        raise ValueError('Invalid "capture_type" option')

    cv2_frame = capture_cv2_frame_func(frame_delay, **capture_options)
    return Calibrator(cv2_frame, frame_stack_func)


def run_tracker(capture_type, laser_tracking_type, landmine_tracking_type,
                frame_delay, calibrator, **options):
    capture_options = {}
    capture_cv2_frame_func = None

    if capture_type == 'webcam':
        capture_cv2_frame_func = ic.capture_webcam_cv2_frame
        capture_options = {
            'video_capture': cv2.VideoCapture(0)
        }
    elif capture_type == 'esp32':
        capture_cv2_frame_func = ic.capture_esp32_cv2_frame
        capture_options = {
            'image_url': options['image_url']
        }
    else:
        raise ValueError('Invalid "capture_type" option')

    laser_tracking_options = {}
    laser_tracking_func = None

    if laser_tracking_type == 'gaussian':
        lower_threshold, upper_threshold = calibrator.calibrate_threshold()
        gaussian_blur_radius = calibrator.calibrate_gaussian_blur()
        laser_tracking_func = tracker.gaussian_tracking
        laser_tracking_options = {
            'screen_name': 'Gaussian Laser Tracking',
            'gaussian_blur_radius': gaussian_blur_radius,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold
        }
    elif laser_tracking_type == 'hsv':
        lower_hsv, upper_hsv = calibrator.calibrate_hsv()
        laser_tracking_func = tracker.hsv_tracking
        laser_tracking_options = {
            'screen_name': 'HSV Laser Tracking',
            'lower_hsv': lower_hsv,
            'upper_hsv': upper_hsv
        }
    else:
        raise ValueError('Invalid "laser.tracking_type" option')

    landmine_tracking_options = {}
    landmine_tracking_func = None

    if landmine_tracking_type == 'hsv_contour':
        lower_hsv, upper_hsv = calibrator.calibrate_hsv()
        landmine_tracking_func = tracker.hsv_contour_tracking
        landmine_tracking_options = {
            'lower_hsv': lower_hsv,
            'upper_hsv': upper_hsv,
            'outline_color': tuple(options['landmine']['post_processing']['outline']['color']),
            'thickness': options['landmine']['post_processing']['outline']['thickness']
        }
    else:
        raise ValueError('Invalid landmine.tracking_type option')

    while True:
        frame = capture_cv2_frame_func(**capture_options)
        max_loc, laser_output, processed_laser_output = tracker.track_object_from_frame(
            frame,
            laser_tracking_func,
            **laser_tracking_options
        )
        coords, landmine_output, processed_landmine_output = tracker.track_objects_from_frame(
            frame,
            landmine_tracking_func,
            laser_output,
            **landmine_tracking_options
        )
        data = compile_data(frame, max_loc, coords, **options)

        output = add_post_processing(landmine_output, data, **options)
        cv2.imshow(laser_tracking_options['screen_name'], output)

        if capture_type == 'esp32':
            data.pop('landmine')
            requests.post(options['control_url'], data=json.dumps(data))
        if ic.check_close_cv2_window():
            break
        time.sleep(frame_delay)


def compile_data(frame, max_loc, landmine_coords, **options):
    height, width, _ = frame.shape
    laser_pos_left, laser_pos_top = max_loc

    center = width / 2
    center_width = options['laser']['center_width_percentage'] * width
    center_left_line = int(center - (center_width / 2))
    center_right_line = int(center + (center_width / 2))

    landmine_coords.sort(key=lambda l: (height - l['top'], abs(center - l['left'])))
    nearest_landmine = landmine_coords[0] if landmine_coords else {'left': 0, 'top': 0}

    return {
        'frame': {
            'width': width,
            'height': height,
        },
        'center': {
            'width': center_width,
            'left_line': center_left_line,
            'right_line': center_right_line
        },
        'laser': {
            'position': {
                'left': laser_pos_left,
                'top': laser_pos_top
            }
        },
        'nearest_landmine': {
            'position': {
                'left': nearest_landmine['left'],
                'top': nearest_landmine['top']
            }
        },
        'landmine': {
            'positions': landmine_coords
        }
    }


def main():
    options = parse_options()
    calibrator = create_calibrator(**options)
    run_tracker(
        laser_tracking_type=options['laser']['tracking_type'],
        landmine_tracking_type=options['landmine']['tracking_type'],
        calibrator=calibrator,
        **options
    )


if __name__ == '__main__':
    main()

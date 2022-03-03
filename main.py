import argparse
import json

from calibrator import Calibrator
from tracker import *
from image_capture import *
from post_processing import add_post_processing


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('-i', '--capture_type', help='["webcam", "esp32"]', default='webcam')
    ap.add_argument('-t', '--tracking_type', help='["gaussian", "hsv"]', default='gaussian')
    # ap.add_argument('-c', '--run_calibrator', action='store_true', default=True)
    ap.add_argument('-d', '--frame_delay', help='frame delay in seconds', type=float, default=0)
    ap.add_argument('-u', '--image_url', help='ESP32 image URL')

    args = vars(ap.parse_args())

    if args['capture_type'] == 'esp32' and args['image_url'] is None:
        raise Exception('Must include "--image_url" argument when "--input_type" is "esp32"')

    return args


def run_calibrator(capture_type, tracking_type, frame_delay, **kwargs):
    calibrator = Calibrator(f'{tracking_type} calibrator')

    options = {}
    capture_cv2_frame_func = None

    if capture_type == 'webcam':
        capture_cv2_frame_func = capture_webcam_cv2_frame_from_stream
        options = {
            'screen_name': 'Webcam Frame Capture',
            'video_capture': cv2.VideoCapture(0)
        }
    elif capture_type == 'esp32':
        capture_cv2_frame_func = capture_esp32_cv2_frame_from_stream
        options = {
            'screen_name': 'ESP32 CAM Frame Capture',
            'image_url': kwargs.get('image_url')
        }
    else:
        raise ValueError('Invalid "capture_type" parameter')

    kwargs.update(options)
    calibrator.capture_frame(capture_cv2_frame_func, frame_delay, **kwargs)
    return calibrator


def run_laser_tracking(capture_type, tracking_type, frame_delay, calibrator, **kwargs):
    options = {}
    get_cv2_frame_func = None
    process_frame_func = None
    screen_name = None

    if capture_type == 'webcam':
        get_cv2_frame_func = capture_webcam_cv2_frame
        options = {'video_capture': cv2.VideoCapture(0)}
    elif capture_type == 'esp32':
        get_cv2_frame_func = capture_esp32_cv2_frame
    else:
        raise ValueError('Invalid "capture_type" parameter')

    kwargs.update(options)

    if tracking_type == 'gaussian':
        calibrator.display_gaussian_blur_calibrator()
        radius = calibrator.get_gaussian_blur_radius()
        process_frame_func = gaussian_processing
        screen_name = 'Gaussian Laser Tracking'
        options = {
            'radius': radius,
            'lower_threshold': 220,
            'upper_threshold': 255
        }
    elif tracking_type == 'hsv':
        calibrator.display_hsv_calibrator()
        low_laser_hsv, high_laser_hsv = calibrator.get_hsv_boundaries()
        process_frame_func = hsv_processing
        screen_name = 'HSV Laser Tracking'
        options = {
            'low_hsv': low_laser_hsv,
            'high_hsv': high_laser_hsv
        }
    else:
        raise ValueError('Invalid "tracking_type" parameter')

    kwargs.update(options)

    while True:
        min_loc, max_loc, output = track_laser(get_cv2_frame_func, process_frame_func, **kwargs)
        output = add_post_processing(output, max_loc)
        cv2.imshow(screen_name, output)
        if kwargs["image_url"] is not None:
            send_control_data(max_loc, output)
        if check_close_cv2_window():
            cv2.destroyWindow(screen_name)
            break
        time.sleep(frame_delay)


def send_control_data(max_loc, output):
    height, width, _ = output.shape
    left, top = max_loc
    data = {
        'frame': {
            'width': width,
            'height': height
        },
        'position': {
            'left': left,
            'top': top
        },
        'in_center': False
    }
    requests.post('http://172.20.10.4/control', data=json.dumps(data))


def main():
    args = parse_args()
    kwargs = {'image_url': args['image_url']}

    calibrator = run_calibrator(
        args['capture_type'],
        args['tracking_type'],
        args['frame_delay'],
        **kwargs
    )

    run_laser_tracking(
        args['capture_type'],
        args['tracking_type'],
        args['frame_delay'],
        calibrator,
        **kwargs
    )


if __name__ == '__main__':
    main()

import json
import yaml

from calibrator import Calibrator
from tracker import *
from image_capture import *
from post_processing import add_post_processing


def parse_args():
    with open('settings.yaml', 'r') as stream:
        args = yaml.safe_load(stream)

    if args['frame_stack'] == 'vertical':
        args['frame_stack_func'] = np.vstack
    elif args['frame_stack'] == 'horizontal':
        args['frame_stack_func'] = np.hstack

    image_url = '{0}://{1}/{2}'.format(
        args['esp32_server']['scheme'],
        args['esp32_server']['host'],
        args['esp32_server']['image_endpoint'],
    )

    control_url = '{0}://{1}/{2}'.format(
        args['esp32_server']['scheme'],
        args['esp32_server']['host'],
        args['esp32_server']['control_endpoint'],
    )

    args['image_url'] = image_url
    args['control_url'] = control_url
    args['frame_delay'] = 1 / args['frames_per_second']

    return args


def run_calibrator(capture_type, laser_tracking_type, frame_delay, **kwargs):
    frame_stack_func = kwargs.get('frame_stack_func')
    calibrator = Calibrator(f'{laser_tracking_type.upper()} calibrator', frame_stack_func)

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


def run_laser_tracking(capture_type, laser_tracking_type, frame_delay, calibrator, **kwargs):
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

    if laser_tracking_type == 'gaussian':
        calibrator.display_threshold_calibrator()
        lower_threshold, upper_threshold = calibrator.get_thresholds()
        calibrator.display_gaussian_blur_calibrator()
        radius = calibrator.get_gaussian_blur_radius()
        process_frame_func = gaussian_processing
        screen_name = 'Gaussian Laser Tracking'
        options = {
            'radius': radius,
            'lower_threshold': lower_threshold,
            'upper_threshold': upper_threshold
        }
    elif laser_tracking_type == 'hsv':
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
    control_url = kwargs.get('control_url')
    frame_stack_func = kwargs.get('frame_stack_func')

    while True:
        min_loc, max_loc, output, processed_output = track_laser(get_cv2_frame_func, process_frame_func, **kwargs)
        output = add_post_processing(output, max_loc)
        cv2.imshow(screen_name, frame_stack_func([output, processed_output]))
        if capture_type == 'esp32':
            send_control_data(max_loc, output, control_url)
        if check_close_cv2_window():
            cv2.destroyWindow(screen_name)
            break
        time.sleep(frame_delay)


def send_control_data(max_loc, output, control_url):
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
    requests.post(control_url, data=json.dumps(data))


def main():
    args = parse_args()
    kwargs = {
        'image_url': args['image_url'],
        'control_url': args['control_url'],
        'frame_stack_func': args['frame_stack_func']
    }

    calibrator = run_calibrator(
        args['capture_type'],
        args['laser_tracking_type'],
        args['frame_delay'],
        **kwargs
    )

    run_laser_tracking(
        args['capture_type'],
        args['laser_tracking_type'],
        args['frame_delay'],
        calibrator,
        **kwargs
    )


if __name__ == '__main__':
    main()

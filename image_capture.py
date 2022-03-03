import cv2
import numpy as np
import requests
import time


class ESP32CAMImageDownloadError(Exception):
    def __init__(self, message='Failed to download image from ESP32 CAM'):
        self.message = message
        super().__init__(self.message)


def capture_webcam_cv2_frame_from_stream(frame_delay, **kwargs):
    screen_name = kwargs.pop('screen_name')
    capture = kwargs.get('video_capture')

    while True:
        _, frame = capture.read()
        cv2.imshow(screen_name, frame)

        if check_close_cv2_window():
            cv2.destroyWindow(screen_name)
            return frame

        time.sleep(frame_delay)


def capture_webcam_cv2_frame(**kwargs):
    capture = kwargs.get('video_capture')
    _, frame = capture.read()
    return frame


def capture_esp32_cv2_frame_from_stream(frame_delay, **kwargs):
    screen_name = kwargs.pop('screen_name')
    image_url = kwargs.get('image_url')

    while True:
        response = requests.get(image_url, stream=True)
        image = None

        if response.ok:
            image_data = np.asarray(bytearray(response.raw.read()), dtype='uint8')
            image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            cv2.imshow(screen_name, image)
        else:
            raise ESP32CAMImageDownloadError

        if check_close_cv2_window():
            cv2.destroyWindow(screen_name)
            return image

        time.sleep(frame_delay)


def capture_esp32_cv2_frame(**kwargs):
    image_url = kwargs.get('image_url')
    response = requests.get(image_url, stream=True)

    if response.ok:
        image_data = np.asarray(bytearray(response.raw.read()), dtype='uint8')
        return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    else:
        raise ESP32CAMImageDownloadError


def check_close_cv2_window():
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True

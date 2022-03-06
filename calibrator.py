import cv2
import numpy as np
import argparse
from enum import Enum


class UpdateType(Enum):
    HUE_LOW = 0
    SAT_LOW = 1
    VAL_LOW = 2
    HUE_HIGH = 3
    SAT_HIGH = 4
    VAL_HIGH = 5
    GAUSSIAN_BLUR = 6
    MIN_THRESHOLD = 7
    MAX_THRESHOLD = 8


class MissingCalibratorFrameError(Exception):
    def __init__(self, message='Calibrator frame does not exist'):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message} -> run capture_calibrator_frame() or upload_calibrator_frame()'


class Calibrator:
    def __init__(self, screen_name, frame_stack_func):
        self.screen_name = screen_name
        self.frame_stack_func = frame_stack_func
        self.original_frame = None

        self.hsv_frame = None
        self.low_hsv = [0, 0, 0]
        self.high_hsv = [255, 255, 255]

        self.gray_frame = None
        self.gaussian_blur_radius = 1

        self.thresh_inv_frame = None
        self.min_threshold = 0
        self.max_threshold = 255

    def get_hsv_boundaries(self):
        return self.low_hsv, self.high_hsv

    def get_gaussian_blur_radius(self):
        return self.gaussian_blur_radius

    def get_thresholds(self):
        return self.min_threshold, self.max_threshold

    def capture_frame(self, capture_cv2_frame_func, frame_delay, **kwargs):
        """
        Sets the frame to perform the calibration on.
        :param frame_delay: The delay between each frame captures
        :param capture_cv2_frame_func: A function that returns a cv2 frame (e.g. cv2.imread(file_name))
        :return: None
        """
        self.original_frame = capture_cv2_frame_func(frame_delay, **kwargs)
        self.hsv_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2HSV)
        self.gray_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2GRAY)
        self.thresh_inv_frame = self.gray_frame.copy()

    def display_hsv_calibrator(self):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        cv2.imshow(self.screen_name, self.frame_stack_func([self.original_frame, self.original_frame]))

        cv2.createTrackbar('HUE [LOW]', self.screen_name, 0, 255, lambda x: self.update_hsv_frame(x, UpdateType.HUE_LOW))
        cv2.createTrackbar('SAT [LOW]', self.screen_name, 0, 255, lambda x: self.update_hsv_frame(x, UpdateType.SAT_LOW))
        cv2.createTrackbar('VAL [LOW]', self.screen_name, 0, 255, lambda x: self.update_hsv_frame(x, UpdateType.VAL_LOW))

        cv2.createTrackbar('HUE [HIGH]', self.screen_name, 255, 255, lambda x: self.update_hsv_frame(x, UpdateType.HUE_HIGH))
        cv2.createTrackbar('SAT [HIGH]', self.screen_name, 255, 255, lambda x: self.update_hsv_frame(x, UpdateType.SAT_HIGH))
        cv2.createTrackbar('VAL [HIGH]', self.screen_name, 255, 255, lambda x: self.update_hsv_frame(x, UpdateType.VAL_HIGH))

        cv2.waitKey(0)
        cv2.destroyWindow(self.screen_name)

    def update_hsv_frame(self, update_value, update_type: UpdateType):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        if update_type == UpdateType.HUE_LOW:
            self.low_hsv[0] = update_value
        elif update_type == UpdateType.SAT_LOW:
            self.low_hsv[1] = update_value
        elif update_type == UpdateType.VAL_LOW:
            self.low_hsv[2] = update_value
        elif update_type == UpdateType.HUE_HIGH:
            self.high_hsv[0] = update_value
        elif update_type == UpdateType.SAT_HIGH:
            self.high_hsv[1] = update_value
        elif update_type == UpdateType.VAL_HIGH:
            self.high_hsv[2] = update_value

        mask = cv2.inRange(self.hsv_frame, np.array(self.low_hsv), np.array(self.high_hsv))
        output = cv2.bitwise_and(self.original_frame, self.original_frame, mask=mask)
        cv2.imshow(self.screen_name, self.frame_stack_func([self.original_frame, output]))

    def display_threshold_calibrator(self):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        cv2.imshow(self.screen_name, self.frame_stack_func([self.gray_frame, self.gray_frame]))
        cv2.createTrackbar('MIN THRESHOLD', self.screen_name, 0, 255, lambda x: self.update_threshold(x, UpdateType.MIN_THRESHOLD))
        cv2.createTrackbar('MAX THRESHOLD', self.screen_name, 255, 255, lambda x: self.update_threshold(x, UpdateType.MAX_THRESHOLD))

        cv2.waitKey(0)
        cv2.destroyWindow(self.screen_name)

    def update_threshold(self, update_value, update_type: UpdateType):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        if update_type == UpdateType.MIN_THRESHOLD:
            self.min_threshold = update_value
        elif update_type == UpdateType.MAX_THRESHOLD:
            self.max_threshold = update_value

        _, thresh_inv = cv2.threshold(
            self.gray_frame,
            self.min_threshold,
            self.max_threshold,
            cv2.THRESH_BINARY
        )

        self.thresh_inv_frame = cv2.bitwise_and(self.gray_frame, thresh_inv)
        cv2.imshow(self.screen_name, self.frame_stack_func([self.gray_frame, self.thresh_inv_frame]))

    def display_gaussian_blur_calibrator(self):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        temp_thresh_inv_frame = cv2.cvtColor(self.thresh_inv_frame.copy(), cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.screen_name, self.frame_stack_func([self.original_frame, temp_thresh_inv_frame]))
        cv2.createTrackbar('RADIUS', self.screen_name, 1, 200, lambda x: self.update_gaussian_blur(x, UpdateType.GAUSSIAN_BLUR))

        cv2.waitKey(0)
        cv2.destroyWindow(self.screen_name)

    def update_gaussian_blur(self, update_value, update_type: UpdateType):
        if self.original_frame is None:
            raise MissingCalibratorFrameError

        if update_type == UpdateType.GAUSSIAN_BLUR and update_value % 2 == 1:
            self.gaussian_blur_radius = update_value

            temp_original_frame = self.original_frame.copy()
            temp_thresh_inv_frame = cv2.cvtColor(self.thresh_inv_frame.copy(), cv2.COLOR_GRAY2BGR)
            temp_blur_frame = cv2.GaussianBlur(self.gray_frame, (update_value, update_value), 0)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_blur_frame)
            cv2.circle(temp_original_frame, max_loc, self.gaussian_blur_radius, (255, 0, 0), 2)
            cv2.circle(temp_thresh_inv_frame, max_loc, self.gaussian_blur_radius, (255, 0, 0), 2)

            cv2.imshow(self.screen_name, self.frame_stack_func([temp_original_frame, temp_thresh_inv_frame]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', help='path to image')
    args = vars(ap.parse_args())

    calibrator = Calibrator('Calibrator', np.hstack)
    calibrator.original_frame = cv2.imread(args['image'])
    calibrator.hsv_frame = cv2.cvtColor(calibrator.original_frame, cv2.COLOR_BGR2HSV)
    calibrator.gray_frame = cv2.cvtColor(calibrator.original_frame, cv2.COLOR_BGR2GRAY)

    calibrator.display_hsv_calibrator()
    low_hsv, high_hsv = calibrator.get_hsv_boundaries()

    calibrator.display_gaussian_blur_calibrator()
    gaussian_blur_radius = calibrator.get_gaussian_blur_radius()

    print(f'low_hsv={low_hsv}')
    print(f'high_hsv={high_hsv}')
    print(f'gaussian_blur_radius={gaussian_blur_radius}')


if __name__ == '__main__':
    main()

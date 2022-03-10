import cv2
import numpy as np

from enum import Enum
from dataclasses import dataclass
from typing import Any, List


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


@dataclass
class HSVCalibratorData:
    hsv_frame: Any
    lower_hsv: List[int]
    upper_hsv: List[int]


@dataclass
class GaussianCalibratorData:
    gray_frame: Any
    gaussian_blur_radius: int


@dataclass
class ThresholdCalibratorData:
    threshold_frame: Any
    lower_threshold: int
    upper_threshold: int


class Calibrator:
    def __init__(self, cv2_frame, frame_stack_func):
        self.hsv_data = HSVCalibratorData(
            hsv_frame=cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2HSV),
            lower_hsv=[0, 0, 0],
            upper_hsv=[255, 255, 255]
        )
        self.gaussian_data = GaussianCalibratorData(
            gray_frame=cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2GRAY),
            gaussian_blur_radius=1
        )
        self.threshold_data = ThresholdCalibratorData(
            threshold_frame=cv2.cvtColor(self.gaussian_data.gray_frame, cv2.COLOR_GRAY2BGR),
            lower_threshold=0,
            upper_threshold=255
        )
        self._frame = cv2_frame
        self._frame_stack_func = frame_stack_func
        self._screen_name = 'Calibrator'
        self._control_screen_name = f'{self._screen_name} controls'

    def calibrate_hsv(self):
        cv2.imshow(self._screen_name, self._frame_stack_func([self._frame, self._frame]))
        cv2.namedWindow(self._control_screen_name)

        cv2.createTrackbar(
            'HUE [L]',
            self._control_screen_name,
            0,
            255,
            lambda val: self._update_hsv(val, UpdateType.HUE_LOW)
        )
        cv2.createTrackbar(
            'SAT [L]',
            self._control_screen_name,
            0,
            255,
            lambda val: self._update_hsv(val, UpdateType.SAT_LOW)
        )
        cv2.createTrackbar(
            'VAL [L]',
            self._control_screen_name,
            0,
            255,
            lambda val: self._update_hsv(val, UpdateType.VAL_LOW)
        )
        cv2.createTrackbar(
            'HUE [H]',
            self._control_screen_name,
            255,
            255,
            lambda val: self._update_hsv(val, UpdateType.HUE_HIGH)
        )
        cv2.createTrackbar(
            'SAT [H]',
            self._control_screen_name,
            255,
            255,
            lambda val: self._update_hsv(val, UpdateType.SAT_HIGH)
        )
        cv2.createTrackbar(
            'VAL [H]',
            self._control_screen_name,
            255,
            255,
            lambda val: self._update_hsv(val, UpdateType.VAL_HIGH)
        )

        cv2.waitKey(0)
        cv2.destroyWindow(self._screen_name)
        cv2.destroyWindow(self._control_screen_name)

        return self.hsv_data.lower_hsv, self.hsv_data.upper_hsv

    def _update_hsv(self, update_value: int, update_type: UpdateType):
        lower_hsv = self.hsv_data.lower_hsv
        upper_hsv = self.hsv_data.upper_hsv

        if update_type == UpdateType.HUE_LOW:
            lower_hsv[0] = update_value
        elif update_type == UpdateType.SAT_LOW:
            lower_hsv[1] = update_value
        elif update_type == UpdateType.VAL_LOW:
            lower_hsv[2] = update_value
        elif update_type == UpdateType.HUE_HIGH:
            upper_hsv[0] = update_value
        elif update_type == UpdateType.SAT_HIGH:
            upper_hsv[1] = update_value
        elif update_type == UpdateType.VAL_HIGH:
            upper_hsv[2] = update_value

        mask = cv2.inRange(self.hsv_data.hsv_frame, np.array(lower_hsv), np.array(upper_hsv))
        output = cv2.bitwise_and(self._frame, self._frame, mask=mask)
        cv2.imshow(self._screen_name, self._frame_stack_func([self._frame, output]))

    def calibrate_threshold(self):
        threshold_frame = self.threshold_data.threshold_frame
        cv2.imshow(self._screen_name, self._frame_stack_func([self._frame, threshold_frame]))
        cv2.namedWindow(self._control_screen_name)

        cv2.createTrackbar(
            'THRESH',
            self._control_screen_name,
            0,
            255,
            lambda val: self._update_threshold(val, UpdateType.MIN_THRESHOLD)
        )

        cv2.waitKey(0)
        cv2.destroyWindow(self._screen_name)
        cv2.destroyWindow(self._control_screen_name)

        return self.threshold_data.lower_threshold, self.threshold_data.upper_threshold

    def _update_threshold(self, update_value: int, update_type: UpdateType):
        if update_type == UpdateType.MIN_THRESHOLD:
            self.threshold_data.lower_threshold = update_value
        elif update_type == UpdateType.MAX_THRESHOLD:
            self.threshold_data.upper_threshold = update_value

        threshold_frame = self.threshold_data.threshold_frame

        _, threshold_frame_temp = cv2.threshold(
            threshold_frame,
            self.threshold_data.lower_threshold,
            self.threshold_data.upper_threshold,
            cv2.THRESH_BINARY
        )

        output = cv2.bitwise_and(threshold_frame, threshold_frame_temp)
        cv2.imshow(self._screen_name, self._frame_stack_func([self._frame, output]))

    def calibrate_gaussian_blur(self):
        threshold_frame = self.threshold_data.threshold_frame
        cv2.imshow(self._screen_name, self._frame_stack_func([self._frame, threshold_frame]))
        cv2.namedWindow(self._control_screen_name)

        cv2.createTrackbar(
            'RADIUS',
            self._control_screen_name,
            1,
            200,
            lambda val: self._update_gaussian_blur(val, UpdateType.GAUSSIAN_BLUR)
        )

        cv2.waitKey(0)
        cv2.destroyWindow(self._screen_name)
        cv2.destroyWindow(self._control_screen_name)

        return self.gaussian_data.gaussian_blur_radius

    def _update_gaussian_blur(self, update_value: int, update_type: UpdateType):
        if update_value % 2 == 0:
            return
        elif update_type == UpdateType.GAUSSIAN_BLUR:
            self.gaussian_data.gaussian_blur_radius = update_value

        temp_frame = self._frame.copy()
        blurred_frame = cv2.GaussianBlur(self.gaussian_data.gray_frame, (update_value, update_value), 0)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred_frame)
        cv2.circle(temp_frame, max_loc, update_value, (255, 0, 0), 2)
        cv2.circle(blurred_frame, max_loc, update_value, (255, 0, 0), 2)

        output = cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self._screen_name, self._frame_stack_func([temp_frame, output]))


def main():
    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    _, frame = capture.read()
    capture.release()

    calibrator = Calibrator(frame, np.vstack)

    lower_hsv, upper_hsv = calibrator.calibrate_hsv()
    print(f'lower_hsv={lower_hsv}')
    print(f'upper_hsv={upper_hsv}')

    lower_threshold, _ = calibrator.calibrate_threshold()
    print(f'lower_threshold={lower_threshold}')

    gaussian_blur_radius = calibrator.calibrate_gaussian_blur()
    print(f'gaussian_blur_radius={gaussian_blur_radius}')


if __name__ == '__main__':
    main()

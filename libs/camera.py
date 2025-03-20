""" ref:
https://github.com/ECI-Robotics/opencv_remote_streaming_processing/
"""

import cv2
import os
from logging import getLogger

logger = getLogger(__name__)


class VideoCamera:
    def __init__(self, input, resize_width, v4l):
        self.resize_width = resize_width
        if input == "cam":
            self.input_stream = 0
            if v4l:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            else:
                self.cap = cv2.VideoCapture(self.input_stream)

        elif input.isdigit():
            # Interpret this as a camera index
            self.input_stream = int(input)
            if v4l:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            else:
                self.cap = cv2.VideoCapture(self.input_stream)

        else:
            self.input_stream = input
            assert os.path.isfile(input), "Specified input file doesn't exist"
            self.cap = cv2.VideoCapture(self.input_stream)

        # 🔥 Force Resolution to 1080p (1920x1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Optional: Improve Smoothness by Setting FPS
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        ret, self.frame = self.cap.read()

        if ret:
            cap_prop = self._get_cap_prop()
            logger.info(
                "cap_pop:{}, resize_width:{}".format(cap_prop, self.resize_width)
            )
        else:
            logger.error(
                "Please try to start with command line parameters using --v4l if you use RaspCamera"
            )
            os._exit(1)

        # Removed resizing logic to ensure full 1080p resolution
        if self.frame.shape[1] != 1920:
            self.frame = cv2.resize(self.frame, (1920, 1080))

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return (
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.cap.get(cv2.CAP_PROP_FPS),
        )

    def get_frame(self, flip_code):

        ret, frame = self.cap.read()

        if frame is None:
            return frame

        # Ensure frame stays at 1080p, no resizing required
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))

        if ret:
            if self.input_stream == 0 and flip_code is not None:
                frame = cv2.flip(frame, int(flip_code))

        return frame

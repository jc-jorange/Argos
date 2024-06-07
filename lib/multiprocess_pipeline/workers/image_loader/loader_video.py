import cv2
import numpy as np
import os
import math
import time

from . import BaseImageLoader


class VideoDataLoader(BaseImageLoader):
    def __init__(self, *args, **kwargs):
        super(VideoDataLoader, self).__init__(*args, **kwargs)
        self.cap = cv2.VideoCapture(self.data_path)
        self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert self.cap.isOpened(), f'Not a valid video on {self.data_path}'

        self.timestamp = []
        self.timestamp_cap = {}
        if os.path.isfile(self.timestamp_path):
            timestamp_format = ['.csv', '.txt']
            if os.path.splitext(self.timestamp_path)[-1] in timestamp_format:
                with open(self.timestamp_path) as f:
                    lines = f.readlines()
                    for e in lines:
                        e = int(float(e.strip()))
                        self.timestamp.append(e)

    def __next__(self):
        super(VideoDataLoader, self).__next__()
        if self.count >= len(self):
            raise StopIteration
        return self.read_image(self.count - 1)

    def read_action(self, idx) -> (int, str, np.ndarray):
        img_path = str(idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        res, img_0 = self.cap.read()   # BGR
        assert img_0 is not None, f'Failed to load frame {img_path}'

        if self.timestamp:
            timestamp = self.timestamp[idx]
        else:
            timestamp = math.floor(time.time() * 1000)
            self.timestamp_cap[idx] = timestamp

        self.image_shape = img_0.shape
        return timestamp, img_path, img_0

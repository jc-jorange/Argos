import math
import os
import glob
import time

import cv2
import numpy as np

from ._masterclass import BaseImageLoader


class ImageDataLoader(BaseImageLoader):
    def __init__(self, *args, **kwargs):
        super(ImageDataLoader, self).__init__(*args, **kwargs)
        self.files = []
        self.timestamp = []
        self.timestamp_cap = {}

        if isinstance(self.data_path, str):
            if os.path.isdir(self.data_path):
                image_format = ['.jpg', '.jpeg', '.png', '.tif', '.exr']
                self.files = sorted(glob.glob('%s/*.*' % self.data_path))
                self.files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, self.files))
            elif os.path.isfile(self.data_path):
                self.files = [self.data_path]
            else:
                raise
        elif isinstance(self.data_path, list):
            self.files = self.data_path

        if os.path.isfile(self.timestamp_path):
            timestamp_format = ['.csv', '.txt']
            if os.path.splitext(self.timestamp_path)[-1] in timestamp_format:
                with open(self.timestamp_path) as f:
                    lines = f.readlines()
                    for e in lines:
                        e = int(float(e.strip()))
                        self.timestamp.append(e)

        self.len = len(self.files)  # number of image files
        assert self.len > 0, f'No images found in {self.data_path}'

    def __next__(self):
        super(ImageDataLoader, self).__next__()
        if self.count == len(self):
            raise StopIteration
        return self.read_image(self.count - 1)

    def read_action(self, idx) -> (int, str, np.ndarray):
        img_path = self.files[idx]
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, f'Failed to load frame {img_path}'

        if self.timestamp:
            timestamp = self.timestamp[idx]
        else:
            timestamp = math.floor(time.time() * 1000)
            self.timestamp_cap[idx] = timestamp

        self.image_shape = img_0.shape
        return timestamp, img_path, img_0

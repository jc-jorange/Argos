import os
import glob
import cv2
import numpy as np

from lib.multiprocess_pipeline.workers.input_data_loader import BaseInputDataLoader


class ImageDataLoader(BaseInputDataLoader):
    def __init__(self, *args):
        super(ImageDataLoader, self).__init__(*args)
        self.files = []
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

        self.len = len(self.files)  # number of image files
        assert self.len > 0, 'No images found in ' + self.data_path

    def __next__(self):
        super(ImageDataLoader, self).__next__()
        if self.count == len(self):
            raise StopIteration
        return self.read_image(self.count - 1)

    def read_action(self, idx) -> (str, np.ndarray):
        img_path = self.files[idx]
        img_0 = cv2.imread(img_path)  # BGR
        assert img_0 is not None, 'Failed to load ' + img_path
        self.image_shape = img_0.shape
        return img_path, img_0

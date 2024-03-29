import cv2
import numpy as np

from lib.input_data_loader import BaseInputDataLoader


class VideoDataLoader(BaseInputDataLoader):
    def __init__(self, *args):
        super(VideoDataLoader, self).__init__(*args)
        self.cap = cv2.VideoCapture(self.data_path)
        self.len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert self.cap.isOpened(), 'Not a valid video for: ' + self.data_path

    def __next__(self):
        super(VideoDataLoader, self).__next__()
        if self.count >= len(self):
            raise StopIteration
        return self.read_image(self.count - 1)

    def read_action(self, idx) -> (str, np.ndarray):
        img_path = str(idx)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        res, img_0 = self.cap.read()   # BGR
        assert img_0 is not None, 'Failed to load frame ' + img_path
        self.image_shape = img_0.shape
        return img_path, img_0

from enum import Enum, unique
import numpy as np
import cv2


@unique
class EInputDataType(Enum):
    Image = 1
    Video = 2
    Address = 3

class BaseInputDataLoader:
    def __init__(self, path: str):
        self.data_path = path

        self.image_shape = (0, 0)
        self.len = 0

        self.count = 0

    def read_image(self, idx) -> (str, np.ndarray, np.ndarray):
        img_path, img_0 = self.read_action(idx)

        # Padded resize
        img, _, _, _ = letterbox(img_0)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0
        return img_path, img, img_0

    def read_action(self, idx) -> (str, np.ndarray):
        return '', np.ndarray

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        return self.read_image(self.count)

    def __getitem__(self, idx):
        idx = idx % len(self)
        return self.read_image(idx)

    def __len__(self):
        return self.len


def letterbox(img,
              height=608,
              width=1088,
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    # padded rectangular
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


from .loader_image import ImageDataLoader
from .loader_video import VideoDataLoader
from .loader_address import AddressDataLoader

loader_factory = {
    EInputDataType.Image.name: ImageDataLoader,
    EInputDataType.Video.name: VideoDataLoader,
    EInputDataType.Address.name: AddressDataLoader,
}
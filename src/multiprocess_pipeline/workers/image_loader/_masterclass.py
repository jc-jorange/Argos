import numpy as np
import cv2


class BaseImageLoader:
    def __init__(self,
                 data_path: str,
                 normalized_image_shape: tuple,
                 timestamp_path=''):
        self.data_path = data_path
        self.timestamp_path = timestamp_path
        self.normalized_image_shape = normalized_image_shape

        self.image_shape = (0, 0)

        self.len = 0
        self.count = 0

    def read_image(self, idx) -> (int, str, np.ndarray, tuple):
        timestamp, img_path, img_0 = self.read_action(idx)

        if img_0 is not None:
            # Padded resize
            img, _, _, _ = letterbox(img_0, self.normalized_image_shape)

            # Normalize RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32)
            img /= 255.0

            return timestamp, img_path, img_0, img

        else:
            raise StopIteration

    def read_action(self, idx) -> (int, str, np.ndarray):
        return 0, '', np.ndarray

    def pre_process(self) -> bool:
        return True

    def __iter__(self):
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
              target_shape=(3, 608, 1088),
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param target_shape:
    :param color:
    :return:
    """
    channels, height, width = target_shape
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

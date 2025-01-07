import numpy as np


class BaseDataFilter:
    def __init__(self,
                 target,
                 *args,
                 **kwargs):
        self.target = target

        self.response = None

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        return data

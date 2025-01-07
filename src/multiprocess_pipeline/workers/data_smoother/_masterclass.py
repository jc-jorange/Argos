import numpy as np


class BaseDataSmoother:
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        self.history_data = None
        self.mask_current = None
        self.mask_history = None

    def put_data(self, data: np.ndarray):
        self.mask_current = np.where(data != 0, 1, 0)
        return

    def cal_history_mask(self):
        if self.history_data is not None:
            self.mask_history = np.where(np.prod(self.history_data, axis=0) != 0, 1, 0)

    def smooth_action(self) -> np.ndarray:
        self.cal_history_mask()
        return None

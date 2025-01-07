import numpy as np

from ._masterclass import BaseDataSmoother


class SimpleMovingAverage(BaseDataSmoother):
    def __init__(self,
                 window_size=5,
                 *args,
                 **kwargs):
        super(SimpleMovingAverage, self).__init__(*args, **kwargs)
        self.window_size = max(1, window_size)

        self.average = None
        self.delta = None

    def put_data(self, data: np.ndarray):
        super(SimpleMovingAverage, self).put_data(data)
        if self.history_data is None:
            self.history_data = np.zeros((self.window_size,)+data.shape)

        self.history_data = np.roll(self.history_data, -1, axis=0)
        self.history_data[-1] = data

    def smooth_action(self) -> np.ndarray:
        super(SimpleMovingAverage, self).smooth_action()

        self.average = np.sum(self.history_data * self.mask_history, axis=0) / self.window_size

        mask_only_current = np.where(self.mask_current - self.mask_history > 0, 1, 0)
        self.average += self.history_data[-1] * mask_only_current

        return self.average

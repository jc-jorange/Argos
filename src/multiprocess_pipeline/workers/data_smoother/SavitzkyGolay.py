import numpy as np
from scipy.signal import savgol_filter

from ._masterclass import BaseDataSmoother


class SavitzkyGolay(BaseDataSmoother):
    def __init__(self,
                 boundary_offset=3,
                 valid_size=11,
                 polyorder=2,
                 *args,
                 **kwargs):
        super(SavitzkyGolay, self).__init__(*args, **kwargs)
        self.polyorder = min(valid_size-1, max(1, polyorder))
        self.boundary_offset = max(1, boundary_offset)
        self.valid_size = max(1, valid_size)

        self.smoothed = None

    def put_data(self, data: np.ndarray):
        super(SavitzkyGolay, self).put_data(data)
        if self.mask_history is None:
            self.history_data = np.zeros((self.valid_size + self.boundary_offset,) + data.shape)

        self.history_data = np.roll(self.history_data, -1, axis=0)
        self.history_data[-1] = data

    def smooth_action(self) -> np.ndarray:
        super(SavitzkyGolay, self).smooth_action()

        self.smoothed = savgol_filter(
            self.history_data * self.mask_history, self.valid_size, self.polyorder, axis=0
        )[-self.boundary_offset]

        mask_only_current = np.where(self.mask_current - self.mask_history > 0, 1, 0)
        self.smoothed += self.history_data[-1] * mask_only_current

        return self.smoothed

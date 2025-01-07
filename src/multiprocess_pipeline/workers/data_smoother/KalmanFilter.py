import numpy as np

from ._masterclass import BaseDataSmoother


class KalmanFilter(BaseDataSmoother):
    def __init__(self,
                 r=0.1**0,
                 q=0.1**0,
                 *args,
                 **kwargs):
        super(KalmanFilter, self).__init__(*args, **kwargs)
        self.R = r
        self.Q = q

        # allocate space for arrays
        self.xhat = None  # a posteri estimate of x
        self.P = None  # a posteri error estimate
        self.xhatminus = None  # a priori estimate of x
        self.Pminus = None  # a priori error estimate
        self.K = None  # gain or blending factor

    def put_data(self, data: np.ndarray):
        super(KalmanFilter, self).put_data(data)
        if self.history_data is None:
            self.history_data = np.zeros((2,) + data.shape)

            self.xhat = np.zeros(data.shape)
            self.P = np.zeros(data.shape)
            self.xhatminus = np.copy(self.xhat)
            self.Pminus = np.copy(self.P)
            self.K = np.zeros(data.shape)

        self.history_data = np.roll(self.history_data, -1, axis=0)
        self.history_data[-1] = data

    def cal_history_mask(self):
        if self.history_data is not None:
            self.mask_history = np.where(self.history_data[0] != 0, 1, 0)

    def smooth_action(self) -> np.ndarray:
        super(KalmanFilter, self).smooth_action()

        self.mask_history_and_current = self.mask_history * self.mask_current

        # predict
        self.xhatminus = self.xhat * self.mask_history_and_current  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1,BU(k) = 0
        self.Pminus = (self.P + self.Q) * self.mask_history_and_current  # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1

        # update
        self.K = self.Pminus / (self.Pminus + self.R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R], H=1
        self.xhat = self.xhatminus + self.K * (
                self.history_data[-1] - self.xhatminus)
        # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        self.P = (1 - self.K) * self.Pminus  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

        mask_only_current = np.where(self.mask_current - self.mask_history > 0, 1, 0)
        self.xhat += self.history_data[-1] * mask_only_current

        return self.xhat

import torch
import numpy as np

from .._masterclass import BasePredictor, S_point


class BaseSpline(BasePredictor):
    m_characteristic = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    m_geometrical_constraint = [
        [S_point],
        [S_point],
        [S_point],
        [S_point],
    ]

    def __init__(self, *args, **kwargs) -> None:
        self.m_characteristic = torch.tensor(self.m_characteristic)
        super(BaseSpline, self).__init__(*args, **kwargs)

    def set_new_base(self, point, t) -> S_point:
        if self.can_process_predict():
            self.process_geometrical_constraint_matrix()
        return super(BaseSpline, self).set_new_base(track_points=point, t=t)

    def clear(self) -> None:
        super(BaseSpline, self).clear()
        self.m_geometrical_constraint = [
            [S_point],
            [S_point],
            [S_point],
            [S_point],
        ]

    def process_geometrical_constraint_matrix(self) -> None:
        p_current = self.p0_list[-1]
        p_previous = self.p0_list[-2]
        d_p = p_current - p_previous
        # d_p = d_p[:, :, 1:2]
        d_p = np.abs(d_p)
        p_current = np.where(d_p < self.max_distance, p_previous, p_current)
        self.p0_list[-1] = p_current

    def predict_content(self, t: float) -> S_point:
        super().predict_content(t)
        self.process_geometrical_constraint_matrix()
        t = ((t - self.time_0) / self.dt_base) + 1
        t = t * 0.01
        m_t = [1, t, t ** 2, t ** 3]
        m_t = torch.tensor(m_t)
        result = m_t @ self.m_characteristic @ self.m_geometrical_constraint
        result = result.numpy()
        return result

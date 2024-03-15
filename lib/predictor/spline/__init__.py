import time
import numpy
import torch

from lib.predictor import BasePredictor, S_point


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

    def __init__(self, **kwargs) -> None:
        self.m_characteristic = torch.tensor(self.m_characteristic)
        super(BaseSpline, self).__init__(**kwargs)

    def set_new_base(self, point) -> S_point:
        self.process_geometrical_constraint_matrix()
        return super(BaseSpline, self).set_new_base(point=point)

    def clear(self) -> None:
        super(BaseSpline, self).clear()
        self.m_geometrical_constraint = [
            [S_point],
            [S_point],
            [S_point],
            [S_point],
        ]

    def process_geometrical_constraint_matrix(self) -> None:
        self.m_geometrical_constraint = torch.tensor(self.m_geometrical_constraint)

    def predict_content(self, t: float) -> torch.Tensor:
        t = ((t - self.time_0) / self.dt_base) * 0.001
        m_t = [1, t, t ** 2, t ** 3]
        m_t = torch.tensor(m_t)
        return m_t @ self.m_characteristic @ self.m_geometrical_constraint

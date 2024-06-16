import torch

from src.multiprocess_pipeline.workers.predictor import BasePredictor, S_point


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

    def set_new_base(self, point) -> S_point:
        if self.can_process_predict():
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

    def predict_content(self, t: float) -> S_point:
        super().predict_content(t)
        self.process_geometrical_constraint_matrix()
        t = ((t - self.time_0) / self.dt_base) * 0.01
        m_t = [1, t, t ** 2, t ** 3]
        m_t = torch.tensor(m_t)
        result = m_t @ self.m_characteristic @ self.m_geometrical_constraint
        result = result.numpy()
        return result

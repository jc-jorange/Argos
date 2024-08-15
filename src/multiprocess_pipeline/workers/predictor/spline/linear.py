import numpy as np
import torch

from ._masterclass import BaseSpline


class LinearSpline(BaseSpline):
    m_characteristic = [
            [1., 0., 0., 0.],
            [-1., 1., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ]

    def can_process_predict(self) -> bool:
        return len(self.p0_list) >= 3

    def process_geometrical_constraint_matrix(self) -> None:
        super(LinearSpline, self).process_geometrical_constraint_matrix()
        raw_1 = torch.tensor(self.p0_list[-2])
        raw_2 = torch.tensor(self.p0_list[-1])

        mask = torch.where((raw_1 * raw_2) > 0.0001, 1., 0.)

        p1 = raw_1 * mask
        p2 = raw_2 * mask

        d_p2 = p2 - p1

        d_t2 = self.time_list[-1] - self.time_list[-2]
        self.dt_base = d_t2

        p1 = p1.unsqueeze(-2)
        p2 = p2.unsqueeze(-2)
        z1 = torch.tensor(np.zeros(p1.shape))
        z2 = torch.tensor(np.zeros(p2.shape))

        self.m_geometrical_constraint = torch.cat((p1,p2,z1,z2), -2).float()

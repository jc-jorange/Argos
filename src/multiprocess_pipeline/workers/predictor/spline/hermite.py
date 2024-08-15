import torch

from ._masterclass import BaseSpline


class HermiteSpline(BaseSpline):
    m_characteristic = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [-3., -2., 3., -1.],
            [2., 1., -2., 1.],
        ]

    def can_process_predict(self) -> bool:
        return len(self.p0_list) >= 3

    def process_geometrical_constraint_matrix(self) -> None:
        super(HermiteSpline, self).process_geometrical_constraint_matrix()
        raw_0 = torch.tensor(self.p0_list[-3])
        raw_1 = torch.tensor(self.p0_list[-2])
        raw_2 = torch.tensor(self.p0_list[-1])

        mask = torch.where((raw_0 * raw_1 * raw_2) > 0.0001, 1., 0.)

        p0 = raw_0 * mask
        p1 = raw_1 * mask
        p2 = raw_2 * mask

        d_p1 = p1 - p0
        d_p2 = p2 - p1

        d_t1 = self.time_list[-2] - self.time_list[-3]
        d_t2 = self.time_list[-1] - self.time_list[-2]
        self.dt_base = d_t2

        v1 = d_p1 / d_t1
        v2 = d_p2 / d_t2

        p1 = p1.unsqueeze(-2)
        p2 = p2.unsqueeze(-2)
        v1 = v1.unsqueeze(-2)
        v2 = v2.unsqueeze(-2)

        self.m_geometrical_constraint = torch.cat((p1,v1,p2,v2), -2).float()

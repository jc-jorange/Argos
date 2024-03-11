import time
from multiprocessing import Process

from lib.predictor.spline.hermite_spline import HermiteSpline
import lib.multiprocess.Shared as Sh
from lib.tracker.utils.utils import *
from lib.tracker.utils import write_result as wr, visualization as vis

class PathPredictProcess(Process):
    m_cha = [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [-3., -2., 3., -1.],
            [2., 1., -2., 1.],
        ]

    def __init__(self,
                 idx: int,
                 opt,
                 shm: Sh.SharedDict
                 ):
        super().__init__()
        self.idx = idx
        self.opt = opt
        self.shm = shm

        self.results_list = []
        self.time_list = []

        # self.m_cha = torch.tensor(self.m_cha).to(self.opt.device).half()
        self.m_cha = torch.tensor(self.m_cha)
        self.m_point = None

        self.t0 = time.perf_counter()
        self.dt_base = 0.00001
        self.bRest = True

    def calculate_path(self, t, m_cha, m_point):
        m_t = [1, t, t ** 2, t ** 3]
        # m_t = torch.tensor(m_t).to(self.opt.device).half()
        m_t = torch.tensor(m_t)

        return m_t@m_cha@m_point

    def process_result(self):
        r0 = torch.tensor(self.results_list[-3])
        r1 = torch.tensor(self.results_list[-2])
        r2 = torch.tensor(self.results_list[-1])

        mask = torch.where((r0 * r1 * r2) > 0.0001, 1., 0.)

        p0 = r0 * mask
        p1 = r1 * mask
        p2 = r2 * mask

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

        return torch.cat((p1,v1,p2,v2), -2).float()

    def reset(self):
        self.t0 = time.perf_counter()
        t2 = self.t0
        if len(self.results_list) >= 3:
            self.results_list.pop(0)
        while len(self.results_list) < 3:
            result = self.pipe_Tracker_read.recv()[:, :, :2]
            self.results_list.append(result)
            recv_time = time.perf_counter()
            self.time_list.append(recv_time)
        # self.m_point = self.prcess_result().to(self.opt.device).half()
        self.m_point = self.process_result()

    def predict(self):
        t1 = time.perf_counter()
        dt = t1 - self.t0
        scaled_dt = dt / self.dt_base
        p_predict = self.calculate_path(scaled_dt, self.m_cha, self.m_point)
        NoneZeroIndex = torch.nonzero(p_predict)

    def run(self):
        t2 = 0
        while True:
            self.bRest = self.pipe_Tracker_read.poll()
            if self.bRest:
                self.reset()

            if len(self.results_list) >= 3:
                t1 = time.perf_counter()
                self.predict()
                dt_each = t1 - t2
                t2 = time.perf_counter()


import numpy as np
import time
import copy

S_point = np.ndarray  # shape:[class, id, (xy)]


class BasePredictor:

    def __init__(self, max_step=16, max_distance=16, *args, **kwargs) -> None:
        self.time_0 = 0.0
        self.dt_base = 1.0
        self.p0_list = []
        self.time_list = []
        self.track_counter = None

        self.max_step = max_step
        self.max_distance = max_distance

    def cull(self, point: S_point, t: float) -> S_point:
        predict_result = self.get_predicted_position(t)
        if not predict_result:
            predict_result = np.zeros(point.shape)
        both = np.multiply(point, predict_result)
        m_p = np.where(both, 0, point)
        m_t = np.where(both, 0, predict_result)

        valid_p = np.nonzero(m_p)
        valid_t = np.nonzero(m_t)

        paired_p = ([], [])
        paired_t = ([], [])

        copy_valid_p = copy.deepcopy(valid_p)
        for i_p in range(len(copy_valid_p[0])):
            class_p = copy_valid_p[0][i_p]
            id_p = copy_valid_p[1][i_p]
            each_p = m_p[class_p, id_p]
            copy_valid_t = copy.deepcopy(valid_t)
            for i_t in range(len(copy_valid_t[0])):
                class_t = copy_valid_t[0][i_t]
                id_t = copy_valid_t[1][i_t]
                each_t = m_t[class_t, id_t]

                dist = np.linalg.norm(each_t - each_p)
                if dist <= self.max_distance:
                    paired_t[0].insert(i_t, valid_t[0][i_t])
                    paired_t[1][i_t] = valid_t[1].pop(i_t)

                    paired_p[0][i_p] = valid_p[0].pop(i_p)
                    paired_p[1][i_p] = valid_p[1].pop(i_p)

                    break

        new_trak = valid_p
        count_trak = valid_t

        return new_trak, count_trak, paired_p, paired_t

    def set_new_base(self, point: S_point) -> None:
        if not self.track_counter:
            self.track_counter = np.where(point[:, :, 0] > 0, 0, -1)
        # self.track_counter = self.track_counter + (point[:, :, 0] > 0)
        t_current = time.perf_counter()
        self.dt_base = t_current - self.time_0
        new_trak, count_trak, paired_p, paired_t = self.cull(point, t_current)
        self.track_counter[new_trak] = 1
        self.track_counter[count_trak] += 1
        point[paired_t] = point[paired_p]
        point[paired_p] = 0

        self.time_0 = t_current
        self.p0_list.append(point)
        self.time_list.append(t_current)

    def clear(self) -> None:
        self.time_0 = 0.0
        self.dt_base = 1.0
        self.p0_list.clear()
        self.time_list.clear()

    def can_process_predict(self) -> bool:
        return True

    def predict_content(self, t: float) -> S_point:
        pass

    def get_predicted_position(self, t: float) -> S_point:
        if self.can_process_predict():
            return self.predict_content(t=t)
        else:
            return None

import numpy as np
import time


S_point = np.ndarray  # shape:[class, id, (xy)]


class BasePredictor:

    def __init__(self,
                 max_step=300,
                 max_distance=50) -> None:
        self.time_0 = 0.0
        self.dt_base = 1.0
        self.p0_list = []
        self.time_list = []
        self.track_counter = None

        self.max_step = max_step
        self.max_distance = max_distance

    def filter_close_track(self, point: S_point, t: float) -> {S_point}:
        predict_result = self.get_predicted_position(t)
        if isinstance(predict_result, np.ndarray):
            predict_result = predict_result
        else:
            predict_result = np.zeros(point.shape)

        both = point * predict_result
        both = np.where(both > 0, 1, 0)
        m_track = np.where(both, 0, point)
        m_predict = np.where(both, 0, predict_result)

        checked_predict = []
        dict_reid = {}  # {(origin_track_class&id) : (predict_class&id)}

        classandid_track = np.nonzero(m_track)
        classandid_predict = np.nonzero(m_predict)

        for i_t in range(len(classandid_track[0])):
            class_t = classandid_track[0][i_t]
            id_t = classandid_track[1][i_t]
            coord_t = m_track[class_t, id_t]
            mm = 2 ** 16
            i_checked = -1
            for i_p in range(len(classandid_predict[0])):
                class_p = classandid_predict[0][i_p]
                id_p = classandid_predict[1][i_p]

                try:
                    if (class_p, id_p) in checked_predict:
                        continue
                except KeyError:
                    pass

                coord_p = m_predict[class_p, id_p]

                dist = np.linalg.norm(coord_p - coord_t)

                if dist <= self.max_distance and dist < mm:
                    dict_reid[(class_t, id_t)] = (class_p, id_p)
                    i_checked = i_p
                    mm = dist
            if i_checked >= 0:
                class_checked = classandid_predict[0][i_checked]
                id_checked = classandid_predict[1][i_checked]
                checked_predict.append((class_checked, id_checked))

        return dict_reid

    def set_new_base(self, point: S_point) -> S_point:
        t_current = time.perf_counter()

        predict_result = self.get_predicted_position(t_current)

        if isinstance(predict_result, np.ndarray):
            reid_track_dict = self.filter_close_track(point, t_current)

            for origin_track_id, new_track_id in reid_track_dict.items():
                point[new_track_id] = point[origin_track_id]
                point[origin_track_id] = 0
            # self.track_counter[np.nonzero(point)] = self.max_step

            # print(numpy.nonzero(predict_result))
            both = point * predict_result
            both = np.where(both > 0, 1, 0)
            # m_track = np.where(both, 0, point)
            m_predict = np.where(both, 0, predict_result)
            point = point + m_predict

            self.time_0 = t_current
            self.p0_list.pop(0)
            self.time_list.pop(0)
            self.p0_list.append(point)
            self.time_list.append(t_current)

            return predict_result
        else:
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

    def get_predicted_position(self, t: float) -> S_point or None:
        if self.can_process_predict():
            return self.predict_content(t=t)
        else:
            return None

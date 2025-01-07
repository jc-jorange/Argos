import numpy as np
import time


S_point = np.ndarray  # shape:[class, id, (xy)]


class BasePredictor:

    def __init__(self,
                 max_step=30,
                 max_distance=50) -> None:
        self.time_0 = 0.0
        self.dt_base = 1.0
        self.p0_list = []
        self.time_list = []
        self.predict_counter = None
        self.mask_keeper = None
        self.last_predict = None

        self.max_step = max(0, max_step)
        self.max_distance = max(0, max_distance)

    def set_new_base(self, track_points: S_point, t) -> S_point:
        # initial counter
        if not isinstance(self.predict_counter, np.ndarray):
            # self.p0_list[-1] += self.p0_list[-2] * self.predict_keeper
            self.predict_counter = np.where(track_points > 0, self.max_step, 0)
            self.mask_keeper = np.zeros(track_points.shape)
            self.last_predict = np.zeros(track_points.shape)

        self.time_0 = t
        self.p0_list.append(track_points)
        self.time_list.append(t)

        # Get normalized tracked and retracked, and non retracked but in keeper
        mask_current = np.where(track_points > 0, 1, 0)
        mask_retracked = (mask_current * self.mask_keeper).astype(np.int32)
        mask_new_tracked = mask_current - mask_retracked
        mask_none_retracked = (self.mask_keeper - mask_retracked).astype(np.int32)
        self.p0_list[-1] += self.last_predict * mask_none_retracked  # copy keep predict

        # Tracked results reset to max step counter
        self.predict_counter[np.nonzero(mask_current)] = self.max_step
        # None retrack counter minus 1
        self.predict_counter -= mask_none_retracked
        # Clamp to 0
        self.predict_counter = np.where(self.predict_counter >= 0, self.predict_counter, 0)
        self.mask_keeper = np.where(self.predict_counter > 0, 1, 0)

        return track_points

    def clear(self) -> None:
        self.time_0 = 0.0
        self.dt_base = 1.0
        self.p0_list.clear()
        self.time_list.clear()
        self.predict_counter = None

    def can_process_predict(self) -> bool:
        return True

    def predict_content(self, t: float) -> S_point:
        pass

    def get_predicted_position(self, t: float) -> S_point or None:
        if self.can_process_predict():
            self.last_predict = self.predict_content(t=t)
            return self.last_predict
        else:
            return None

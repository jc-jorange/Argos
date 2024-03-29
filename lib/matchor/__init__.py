import numpy as np
import time

S_Match_point = np.ndarray  # shape:[class, id, (xy)]


class BaseMatchor:
    def __init__(self,
                 intrinsic_parameters_dict,
                 max_range=10000,
                 max_distance=100):
        super().__init__()

        # intrinsic_parameters_dict = {camera_key: Intrinsic Parameters Numpy Matrix}

        self.camera_position_dict = {}
        self.intrinsic_parameters_dict = intrinsic_parameters_dict

        self.baseline_result = None
        self.baseline_result_in_camera = None
        self.baseline_camera_position = np.empty(4)

        self.match_result = None

        self.max_range = max_range
        self.max_distance = max_distance

    def get_baseline_result(self) -> S_Match_point:
        ...

    def match_content(self, idx: int, predict_result: S_Match_point) -> S_Match_point:
        ...

    def get_match_result(self, camera_id, predict_result) -> S_Match_point:
        if not isinstance(self.baseline_result, S_Match_point):
            self.baseline_result_in_camera = self.get_baseline_result()

        return self.match_content(camera_id, predict_result)

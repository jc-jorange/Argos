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

        self.unmatch_result_dict = {}
        self.camera_position_dict = {}
        self.intrinsic_parameters_dict = intrinsic_parameters_dict

        self.baseline_result = None
        self.baseline_camera_position = np.empty(4)

        self.time_0 = 0.0

        self.match_result = None

        self.max_range = max_range
        self.max_distance = max_distance

    def get_baseline_result(self) -> S_Match_point:
        ...

    def match_content(self, idx: int, key: any, predict_result: S_Match_point) -> None:
        ...

    def set_unmatch_result_and_cameras(self, unmatch_result_dict: {}, camera_position_dict: {}) -> None:
        self.unmatch_result_dict = unmatch_result_dict
        self.camera_position_dict = camera_position_dict

        self.time_0 = time.perf_counter()

    def post_match(self) -> None:
        ...

    def get_match_result(self) -> S_Match_point:
        if not isinstance(self.baseline_result, S_Match_point):
            self.baseline_result = self.get_baseline_result()
        for i, key_unmatch, unmatch_result in enumerate(self.unmatch_result_dict.items()):
            self.match_content(i, key_unmatch, unmatch_result)

        self.post_match()
        return self.match_result

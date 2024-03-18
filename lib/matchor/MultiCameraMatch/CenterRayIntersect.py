from .. import S_Match_point
from ..MultiCameraMatch import BaseMultiCameraMatchor

import numpy as np


class CenterRayIntersectMatchor(BaseMultiCameraMatchor):
    def __init__(self, *args):
        super().__init__(*args)
        self.ray_dict = {}

    @staticmethod
    def get_intersect_t(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> (float, float):
        p2_minus_p1 = p2 - p1
        d2_corss_d1 = np.cross(d1, d2)
        ord_squar_2 = np.linalg.norm(d2_corss_d1) ** 2
        t1 = np.dot(np.cross(p2_minus_p1, d1), d2_corss_d1) / ord_squar_2
        t2 = np.dot(np.cross(p2_minus_p1, d2), d2_corss_d1) / ord_squar_2
        return t1, t2

    @staticmethod
    def get_ray_position(p: np.ndarray, d: np.ndarray, t: float) -> np.ndarray:
        return p + (t * d)

    def match_content(self, idx: int, key: any, predict_result: S_Match_point) -> None:
        ...

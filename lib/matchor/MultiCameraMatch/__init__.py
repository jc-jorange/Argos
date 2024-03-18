import numpy as np

from lib.matchor import BaseMatchor, S_Match_point


class BaseMultiCameraMatchor(BaseMatchor):

    @staticmethod
    def get_point_in_camera_coord(k: np.ndarray, p: np.ndarray) -> np.ndarray:
        # p:[x, y]
        k = np.matrix(k)
        p = np.concatenate((p.T, np.array([[1]])))  # concatenate [u,v]T to [u,v,1]T
        return k.I * p

    def get_baseline_result(self) -> S_Match_point:
        super().get_baseline_result()
        first_key = self.unmatch_result_dict.keys()[0]
        baseline_result: np.ndarray = self.unmatch_result_dict[first_key]
        self.baseline_camera_position = self.camera_position_dict[first_key]

        baseline_intrinsic_parameters = self.intrinsic_parameters_dict[first_key]
        baseline_result_in_camera_coord = np.zeros(
            (baseline_result.shape[0], baseline_result.shape[1], baseline_result.shape[2] + 2)  # [class,id,[x,y,z,1]]
        )

        classandid_baseline_result = np.nonzero(baseline_result)

        for i_b in range(len(classandid_baseline_result[0])):
            class_b = classandid_baseline_result[0][i_b]
            id_b = classandid_baseline_result[1][i_b]
            coord_b = baseline_result[class_b, id_b]

            coord_b_in_camera_coord = self.get_point_in_camera_coord(baseline_intrinsic_parameters, coord_b)
            baseline_result_in_camera_coord[class_b, id_b] = coord_b_in_camera_coord.T

        return baseline_result_in_camera_coord

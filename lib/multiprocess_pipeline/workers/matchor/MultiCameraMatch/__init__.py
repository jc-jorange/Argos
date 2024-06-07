import numpy as np

from lib.multiprocess_pipeline.workers.matchor import BaseMatchor, S_Match_point


class BaseMultiCameraMatchor(BaseMatchor):

    @staticmethod
    def get_point_in_camera_coord(k: np.ndarray, p: np.ndarray) -> np.ndarray:
        # p:[x, y]
        k = np.matrix(k)
        k_i = k.I
        p = p[:, np.newaxis]
        p = np.concatenate((p, np.array([[1]])))  # concatenate [u,v]T to [u,v,1]T
        return k_i @ p

    @staticmethod
    def get_point_in_world_coord(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        p[-1] = 1.0
        t = np.matrix(t)
        t_i = t.I
        return t_i @ p

    def convert_predict_to_camera_coord(self, camera_name, predict_result):
        intrinsic_parameters = self.intrinsic_parameters_dict[camera_name]
        camera_transform = self.camera_transform_dict[camera_name]
        predict_result_in_camera_coord = np.zeros(
            (predict_result.shape[0], predict_result.shape[1], predict_result.shape[2])  # [class,id,[x,y,z,1]]
        )

        classandid_result = np.nonzero(predict_result)

        for i in range(len(classandid_result[0]) // 4):
            class_predict = classandid_result[0][i * 4]
            id_predict = classandid_result[1][i * 4]
            coord_predict = predict_result[class_predict, id_predict][0:2]

            coord_b_in_camera_coord = self.get_point_in_camera_coord(intrinsic_parameters, coord_predict)
            coord_b_in_world_coord = self.get_point_in_world_coord(camera_transform, coord_b_in_camera_coord)
            predict_result_in_camera_coord[class_predict, id_predict] = np.squeeze(coord_b_in_world_coord.T)
            # print(f'idx: {camera_id} ', f'class: {class_predict} ', f'id: {id_predict} ', f'coord: {coord_b_in_camera_coord}')

        return predict_result_in_camera_coord

    def get_baseline_result(self) -> S_Match_point:
        super().get_baseline_result()
        baseline_name = list(self.camera_transform_dict.keys())[0]
        result = self.convert_predict_to_camera_coord(baseline_name, self.baseline_result)
        return result

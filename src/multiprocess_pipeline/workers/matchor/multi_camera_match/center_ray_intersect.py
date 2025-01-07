import numpy

from ._masterclass import S_Match_point
from ._masterclass import BaseMultiCameraMatchor

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

    def match_content(self, name, predict_result: S_Match_point) -> (S_Match_point, S_Match_point):
        # Initialize global 3D position
        global_position = np.copy(predict_result)
        global_position.fill(0)

        # Get target position result and baseline position result in camera coordinate
        predict_result_in_camera_coord = self.convert_predict_to_world_coord(name, predict_result)
        class_id_predict_result = np.nonzero(predict_result)
        class_id_baseline = np.nonzero(self.baseline_result)

        # Get target camera transform and baseline camera transform
        camera_transform_baseline = self.baseline_camera_transform
        camera_transform_predict = self.camera_transform_dict[name]

        camera_position_baseline = camera_transform_baseline[:, 3][0:3]
        # print(f'CAMERA BASE: {camera_position_baseline}')
        camera_position_predict = camera_transform_predict[:, 3][0:3]
        # print(f'CAMERA PREDICT: {camera_position_predict}')

        # Initialize already matched object dirct
        matched_baseline_classandid = {}

        # For each target predict object
        for i_predict in range(len(class_id_predict_result[0]) // 4):
            class_predict = class_id_predict_result[0][i_predict * 4]
            id_predict = class_id_predict_result[1][i_predict * 4]
            world_coord_predict = predict_result_in_camera_coord[class_predict, id_predict][0:3]

            class_base = -1
            id_base = -1

            # Initialize a really far distance as minimum distance
            min_distance_tuple = (-1, -1, 1000000, np.array([0., 0., 0., 0.]))
            # For each baseline object
            for i_base in range(len(class_id_baseline[0]) // 4):
                class_base = class_id_baseline[0][i_base * 4]
                id_base = class_id_baseline[1][i_base * 4]

                # # if this object has already matched, pass
                # if class_base in matched_baseline_classandid.keys():
                #     if id_base == matched_baseline_classandid[class_base]:
                #         continue

                world_coord_base = self.baseline_result_in_camera[class_base, id_base][0:3]

                d_base = world_coord_base - camera_position_baseline
                d_predict = world_coord_predict - camera_position_predict
                # Get t value in each ray line at the closest position
                t1, t2 = self.get_intersect_t(
                    camera_position_baseline, d_base, camera_position_predict, d_predict
                )
                # Get the global position in each ray line at t
                p1 = self.get_ray_position(camera_position_baseline, d_base, t1)
                p2 = self.get_ray_position(camera_position_predict, d_predict, t2)
                # print(f'BASE:  class:{class_base},id:{id_base},coord:{world_coord_base},position:{p1},t:{t1}')
                # print(f'{name}:class:{class_predict},id:{id_predict},coord:{world_coord_predict},position:{p2},t:{t2}')

                # Get the global distance between target and baseline
                distance = np.linalg.norm(p2 - p1)
                # print(distance)

                # if distance smaller than threshold, we first compare it with last minimum distance, if it is lower
                # we save it as new matched pair.
                if distance < self.threshold and 0 < t1 < self.max_range and 0 < t2 < self.max_range:
                    if distance < min_distance_tuple[2]:
                        min_distance_tuple = (class_base, id_base, distance, np.append(p1, 1))

            # print(min_distance_tuple)
            # If we have a matched pair, target object get the baseline object class and id and clear origin
            if min_distance_tuple[0] > -1:
                if class_predict != min_distance_tuple[0] or id_predict != min_distance_tuple[1]:
                    predict_result[min_distance_tuple[0], min_distance_tuple[1]] = predict_result[class_predict,
                                                                                                  id_predict]
                    predict_result[class_predict, id_predict, :] = 0

                # Save this object 3D position
                global_position[min_distance_tuple[0], min_distance_tuple[1]] = min_distance_tuple[3]

                # Update this object into already matched object dirct
                matched_baseline_classandid.update({min_distance_tuple[0]: min_distance_tuple[1]})
            else:
                # If this object in target predict result is not matched but has a same id in baseline, we give it a new
                # id to this object
                if class_predict in class_id_baseline[0]:
                    if id_predict == class_id_baseline[1][list(class_id_baseline[0]).index(class_predict)]:

                        # from current id, add 1 until find a none used id.
                        id_loop = id_predict
                        d_i = 1
                        while (id_loop + d_i) in class_id_predict_result[1]:
                            # If loop to the end, from 0 and start
                            if id_loop + d_i > predict_result.shape[1]:
                                id_loop = 0
                                d_i = 0
                            else:
                                d_i += 1

                        new_id = id_loop + d_i
                        predict_result[class_base, new_id] = predict_result[class_predict, id_predict]
                        self.baseline_result[class_base, new_id] = predict_result[class_predict, id_predict]

                        predict_result[class_predict, id_predict, :] = 0

                        # Set this unmatched object new id as new object into baseline
                        self.baseline_result_in_camera[class_base, new_id] = \
                            predict_result_in_camera_coord[class_predict, id_predict]

                    # Set this unmatched object as new object into baseline
                    else:
                        self.baseline_result[class_predict, id_predict] = predict_result[class_predict, id_predict]
                        self.baseline_result_in_camera[class_predict, id_predict] = \
                            predict_result_in_camera_coord[class_predict, id_predict]
                # Set this unmatched object as new object into baseline
                else:
                    self.baseline_result[class_predict, id_predict] = predict_result[class_predict, id_predict]
                    self.baseline_result_in_camera[class_predict, id_predict] = \
                        predict_result_in_camera_coord[class_predict, id_predict]

        # print(numpy.nonzero(predict_result))
        # print(numpy.nonzero(global_position))
        return predict_result, global_position

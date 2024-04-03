import multiprocessing
import time
from typing import Type
from collections import defaultdict
import numpy

from ..multiprocess import BaseProcess
from .SharedMemory import EQueueType
from lib.matchor import BaseMatchor
from ..postprocess.utils import write_result as wr

# Test_Camera = {
#             2: numpy.array([
#                 [-0.027821],
#                 [-0.70655],
#                 [-0.250505],
#                 [1],
#             ]).T,
#             0: numpy.array([
#                 [0],
#                 [-0.69785],
#                 [-0.268505],
#                 [1],
#             ]).T,
#             1: numpy.array([
#                 [0.027821],
#                 [-0.70655],
#                 [-0.250505],
#                 [1]
#             ]).T,
# }

Test_Camera = {
            2: numpy.array([
                [0.15],
                [1.25],
                [0],
                [1],
            ]).T,
            0: numpy.array([
                [0],
                [1],
                [0],
                [1],
            ]).T,
            1: numpy.array([
                [0.1],
                [0.95],
                [0],
                [1]
            ]).T,
}

intrinsic_parameters_dict = {
    0: [
        [445.2176, 0.6986, 349.3952, 0],
        [0, 444.1691, 214.1200, 0],
        [0, 0, 1.0000, 0],
    ],
    1: [
        [445.2176, 0.6986, 349.3952, 0],
        [0, 444.1691, 214.1200, 0],
        [0, 0, 1.0000, 0],
    ],
    2: [
        [445.2176, 0.6986, 349.3952, 0],
        [0, 444.1691, 214.1200, 0],
        [0, 0, 1.0000, 0],
    ],
}


class GlobalIdMatchProcess(BaseProcess):
    prefix = 'Argus-SubProcess-Global_ID_Match_'
    dir_name = 'id_match'
    log_name = 'ID_Match_Log'
    save_type = [wr.E_text_result_type.raw]

    def __init__(self,
                 queue_dict: {},
                 matchor: Type[BaseMatchor],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.queue_dict = queue_dict

        self.match_result_dir_dict = {}
        for i, queue in queue_dict.items():
            self.match_result_dir_dict[i] = self.making_dir(self.main_output_dir, str(i+1))

        self.matchor = matchor(intrinsic_parameters_dict)

    def match_other_predict(self, frame, subframe, camera_i, predict_result):
        t1 = time.perf_counter()
        result = self.matchor.get_match_result(camera_i, predict_result)
        t2 = time.perf_counter()
        fps = 1 / (t2 - t1)

        result_frame = {}
        result_each_subframe = {}
        result_class = defaultdict(dict)
        result_id = {}
        valid_position = numpy.nonzero(result)
        target_num = len(valid_position[0]) // 4
        for i in range(target_num):
            cls = valid_position[0][i * 4]
            target_id = valid_position[1][i * 4]
            x_position = valid_position[2][(i * 4)]
            y_position = valid_position[2][(i * 4) + 1]
            x = result[cls][target_id][x_position]
            y = result[cls][target_id][y_position]
            result_id[target_id] = ((x, y, 0, 0), 1.0)
            result_class[cls].update(result_id)

        result_each_subframe[subframe] = (result_class, fps)
        result_frame[frame] = result_each_subframe

        save_dir = self.match_result_dir_dict[camera_i]

        self.save_result_to_file(save_dir, result_frame)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start global matching')
        # predict_queue = self.shared_container.queue_dict[EQueueType.PredictResultSend]
        self.matchor.camera_position_dict = Test_Camera
        while self.shared_container.b_input_loading.value:
            for i_camera, each_queue in self.queue_dict.items():
                if i_camera == 0:
                    try:
                        predict_result = each_queue.get(block=False)
                        frame = predict_result[0]
                        subframe = predict_result[1]
                        result = predict_result[2]
                        self.matchor.baseline_result = predict_result[2]
                    except multiprocessing.queues.Empty:
                        break
                    self.matchor.baseline_camera_position = Test_Camera[0]

                    result_frame = {}
                    result_each_subframe = {}
                    result_class = defaultdict(dict)
                    result_id = {}
                    valid_position = numpy.nonzero(result)
                    target_num = len(valid_position[0]) // 4
                    for i in range(target_num):
                        cls = valid_position[0][i * 4]
                        target_id = valid_position[1][i * 4]
                        x_position = valid_position[2][(i * 4)]
                        y_position = valid_position[2][(i * 4) + 1]
                        x = result[cls][target_id][x_position]
                        y = result[cls][target_id][y_position]
                        # result_id[target_id] = ((x, y, 0, 0), 1.0)
                        result_class[cls][target_id] = ((x, y, 0, 0), 1.0)

                    result_each_subframe[subframe] = (result_class, 0.0)
                    result_frame[frame] = result_each_subframe

                    save_dir = self.match_result_dir_dict[0]

                    self.save_result_to_file(save_dir, result_frame)

                else:
                    try:
                        each_result = each_queue.get(block=False)
                        frame = each_result[0]
                        subframe = each_result[1]
                        each_result = each_result[2]
                        if isinstance(each_result, numpy.ndarray) and isinstance(self.matchor.baseline_result, numpy.ndarray):
                            self.match_other_predict(frame, subframe, i_camera, each_result)
                    except multiprocessing.queues.Empty:
                        continue

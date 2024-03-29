import multiprocessing
from typing import Type

import numpy

from ..multiprocess import BaseProcess
from .SharedMemory import EQueueType
from lib.matchor import BaseMatchor
from lib.matchor.MultiCameraMatch import CenterRayIntersect

Test_Camera = {
            0: numpy.array([
                [-0.027821],
                [-0.70655],
                [0.250505],
                [1],
            ]).T,
            1: numpy.array([
                [0],
                [-0.69785],
                [0.268505],
                [1],
            ]).T,
            2: numpy.array([
                [0.027821],
                [-0.70655],
                [0.250505],
                [1]
            ]).T,
}

intrinsic_parameters_dict = {
    0: [
        [445.2176, 0.6986, 349.3952],
        [0, 444.1691, 214.1200],
        [0, 0, 1.0000],
    ],
    1: [
        [445.2176, 0.6986, 349.3952],
        [0, 444.1691, 214.1200],
        [0, 0, 1.0000],
    ],
    2: [
        [445.2176, 0.6986, 349.3952],
        [0, 444.1691, 214.1200],
        [0, 0, 1.0000],
    ],
}


class GlobalIdMatchProcess(BaseProcess):
    def __init__(self,
                 queue_dict: {},
                 matchor: Type[BaseMatchor],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.queue_dict = queue_dict

        self.matchor = matchor(intrinsic_parameters_dict)

    def match_other_predict(self, camera_i, predict_result):
        self.matchor.get_match_result(camera_i, predict_result)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start global matching')
        # predict_queue = self.shared_container.queue_dict[EQueueType.PredictResultSend]
        self.matchor.camera_position_dict = Test_Camera
        while self.shared_container.b_input_loading.value:
            for i_camera, each_queue in self.queue_dict.items():
                if i_camera == 0:
                    try:
                        self.matchor.baseline_result = each_queue.get(block=False)
                    except multiprocessing.queues.Empty:
                        break
                    self.matchor.baseline_camera_position = Test_Camera[0]
                else:
                    try:
                        self.match_other_predict(i_camera, each_queue.get(block=False))
                    except multiprocessing.queues.Empty:
                        break

from typing import Type

from ..multiprocess import BaseProcess, EMultiprocess
from Main import EQueueType
from lib.matchor import BaseMatchor
from lib.matchor.MultiCameraMatch import CenterRayIntersect


class GlobalIdMatchProcess(BaseProcess):
    def __init__(self,
                 matchor: Type[BaseMatchor],
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
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
        self.matchor = matchor(intrinsic_parameters_dict)

    def get_base_ids(self):
        self.matchor.set_unmatch_result_and_cameras()

    def match_other_predict(self):
        ...

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start global matching')
        predict_queue = self.container_queue[EQueueType.PredictToGlobalMatch]
        while not self.end_run_flag.value:
            for i_camera, each_camera_predict in predict_queue.items():
                if i_camera == 0:
                    self.get_base_ids()
                else:
                    self.match_other_predict()

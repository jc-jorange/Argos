import multiprocessing
import time
from typing import Type
from collections import defaultdict
import numpy

from lib.multiprocess import ConsumerProcess
from lib.multiprocess.SharedMemory import E_ProducerOutputName_Global, E_ProducerOutputName_Global_PassThrough, E_SharedSaveType
from lib.multiprocess.SharedMemory import ProducerHub_Global
from lib.matchor import BaseMatchor
from lib.postprocess.utils.write_result import convert_numpy_to_dict
from lib.postprocess.utils import write_result as wr
from lib.matchor.MultiCameraMatch.CenterRayIntersect import CenterRayIntersectMatchor

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


class MultiCameraIdMatchProcess(ConsumerProcess):
    prefix = 'Argus-SubProcess-Global_ID_Match_'
    dir_name = 'id_match'
    log_name = 'ID_Match_Log'
    save_type = [wr.E_text_result_type.raw]

    def __init__(self,
                 matchor: Type[BaseMatchor] = CenterRayIntersectMatchor,
                 output_type=E_SharedSaveType.Queue,
                 output_shape=(1,),
                 *args,
                 **kwargs):
        super().__init__(output_type, output_shape, *args, **kwargs)

        self.producer_result_hub: ProducerHub_Global
        self.indi_result_queue_dict = self.producer_result_hub.output_passthrough

        self.match_result_dir_dict = {}
        for i, queue in self.indi_result_queue_dict.items():
            self.match_result_dir_dict[i] = self.making_dir(self.main_save_dir, str(i + 1))

        self.matchor = matchor(intrinsic_parameters_dict)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start global_process matching')
        self.matchor.camera_position_dict = Test_Camera
        result = numpy.empty((2, 2, 2))
        while self.producer_result_hub.output[E_ProducerOutputName_Global.bInputLoading].b_input_loading.value:
            t1 = time.perf_counter()
            match_times = 0
            for i_camera, each_pass in self.indi_result_queue_dict.items():
                each_queue = each_pass[E_ProducerOutputName_Global_PassThrough.PredictAll]
                match_times += 1
                if i_camera == 0:
                    try:
                        predict_result = each_queue.get(block=False)
                        frame = predict_result[0]
                        subframe = 0
                        result = predict_result[2]
                        self.matchor.baseline_result = predict_result[2]
                    except multiprocessing.queues.Empty:
                        break
                    self.matchor.baseline_camera_position = Test_Camera[0]
                else:
                    try:
                        each_result = each_queue.get(block=False)
                        frame = each_result[0]
                        subframe = each_result[1]
                        each_result = each_result[2]
                        if isinstance(each_result, numpy.ndarray):
                            result = self.matchor.get_match_result(i_camera, each_result)
                    except multiprocessing.queues.Empty:
                        continue

                t2 = time.perf_counter()
                fps = match_times / (t2 - t1)
                result_frame = convert_numpy_to_dict(result, frame, subframe, fps)

                save_dir = self.match_result_dir_dict[i_camera]

                self.save_result_to_file(save_dir, result_frame)

import multiprocessing
import time
import numpy

from lib.multiprocess_pipeline import ConsumerProcess
from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType
from lib.multiprocess_pipeline.SharedMemory import ConsumerOutputPort
from lib.multiprocess_pipeline.workers.matchor import factory_matchor
from lib.multiprocess_pipeline.workers.postprocess.utils.write_result import convert_numpy_to_dict
from lib.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from lib.multiprocess_pipeline.workers.matchor.MultiCameraMatch.CenterRayIntersect import CenterRayIntersectMatchor

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

    output_type = E_SharedSaveType.Queue
    output_shape = (1,)

    def __init__(self,
                 matchor_name: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.indi_port_dict = self.data_hub.consumer_port

        self.match_result_dir_dict = {}
        for i, l in self.indi_port_dict.items():
            self.match_result_dir_dict[i] = self.making_dir(self.results_save_dir, str(i))

        self.matchor = factory_matchor[matchor_name](intrinsic_parameters_dict)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start global_process matching')
        self.matchor.camera_position_dict = Test_Camera
        result = numpy.empty((2, 2, 2))

        hub_b_loading = self.data_hub.bInputLoading[self.idx]

        while hub_b_loading.value:
            t1 = time.perf_counter()
            match_times = 0
            for i_camera, each_pass in self.indi_port_dict.items():
                final_result_port: ConsumerOutputPort = each_pass[-1]
                match_times += 1
                if i_camera == 0:
                    try:
                        predict_result = final_result_port.read()
                        frame = predict_result[0]
                        subframe = 0
                        result = predict_result[2]
                        self.matchor.baseline_result = predict_result[2]
                    except multiprocessing.queues.Empty:
                        break
                    self.matchor.baseline_camera_position = Test_Camera[0]
                else:
                    try:
                        each_result = final_result_port.read()
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

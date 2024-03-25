import time
from typing import Type

from ..multiprocess import BaseProcess, EMultiprocess
from lib.predictor import BasePredictor
from lib.tracker.utils.utils import *
from Main import EQueueType
from lib.postprocess.utils.write_result import Total_Result_Format, Track_Result_Format


class PathPredictProcess(BaseProcess):
    prefix = 'Argus-SubProcess-PathPredictProcess_'
    dir_name = 'predict'

    def __init__(self,
                 predictor_class: Type[BasePredictor],
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.current_track_result = None
        self.current_predict_result = None
        self.all_predict_result = Total_Result_Format

        self.predictor = predictor_class()

    def run_begin(self) -> None:
        super().run_begin()
        self.set_logger_file_handler(self.name + '_PathPredict_Log', self.main_output_dir)
        self.logger.info("This is the Path Predictor Process No.{:d}".format(self.idx))

    def run_action(self) -> None:
        self.logger.info('Start predicting')
        super(PathPredictProcess, self).run_action()
        self.predictor.time_0 = time.perf_counter()
        frame = 0
        subframe = -1
        track_queue = self.container_queue[EQueueType.TrackerToPredict][self.idx]
        predict_queue = self.container_queue[EQueueType.PredictToGlobalMatch][self.idx]
        while not self.end_run_flag.value:
            t1 = time.perf_counter()
            try:
                self.current_track_result = track_queue.get(block=False)
                frame += 1
                subframe = 0
                self.current_predict_result = self.predictor.set_new_base(self.current_track_result)
            except:
                subframe += 1
                self.current_predict_result = self.predictor.get_predicted_position(time.perf_counter())
            t2 = time.perf_counter()
            fps = 1/(t2-t1)

            predict_queue.put(self.current_predict_result)

            if isinstance(self.current_predict_result, np.ndarray):
                for i_c, e_c in enumerate(self.current_predict_result):
                    for i_id, e_id in enumerate(e_c):
                        if e_id[0] > 0 and e_id[2] > 0:
                            e_id = e_id.tolist()
                            result = Track_Result_Format
                            result[i_c][i_id] = ((int(e_id[0]), int(e_id[1])), 1.0)
                            self.all_predict_result[frame][subframe] = [result, fps]

    def run_end(self) -> None:
        super().run_end()

        self.container_result[EMultiprocess.Predictor][self.idx] = self.all_predict_result

        self.logger.info('-'*5 + 'Predict Finished' + '-'*5)

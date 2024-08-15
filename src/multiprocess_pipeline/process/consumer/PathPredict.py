import time
from multiprocessing import queues
from collections import defaultdict

from . import ConsumerProcess
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType
from src.multiprocess_pipeline.workers.predictor import factory_predictor
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from src.multiprocess_pipeline.workers.tracker.utils.utils import *


class PathPredictProcess(ConsumerProcess):
    prefix = 'Argos-SubProcess-PathPredictProcess_'
    dir_name = 'predict'
    log_name = 'Path_Predict_Log'
    save_type = [wr.E_text_result_type.raw]

    output_type = E_SharedSaveType.Queue
    output_data_type = E_OutputPortDataType.CameraTrack
    output_shape = (1,)

    def __init__(self,
                 predictor_name: str,
                 *args,
                 max_step=30,
                 max_distance=5,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.predictor_name = predictor_name
        self.max_step = max_step
        self.max_distance = max_distance

        if self.last_process_port.data_type != E_OutputPortDataType.CameraTrack:
            raise TypeError('Connect last consumer process output data type not fit')

        self.current_track_result = None
        self.current_predict_result = None
        self.all_predict_result = {}

        self.predictor = None

    def run_begin(self) -> None:
        super(PathPredictProcess, self).run_begin()

        self.logger.info(f'Creating predictor {self.predictor_name}')
        self.predictor = factory_predictor[self.predictor_name](
            max_step=self.max_step,
            max_distance=self.max_distance
        )

    def run_action(self) -> None:
        super(PathPredictProcess, self).run_action()
        self.logger.info('Start predicting')

        self.predictor.time_0 = time.perf_counter()
        frame_get = 0
        frame = 0
        subframe = 0

        t_frame_start = time.perf_counter()

        # Get this pipeline producer is alive
        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        while hub_b_loading.value:
            b_get_new_data = True
            t_subframe_start = time.perf_counter()
            try:
                # Get track result from last consumer
                track_result = self.last_process_port.read()
                frame_get = track_result[0]
                self.current_track_result = track_result[-1]
                self.logger.debug(f'Get track result @ frame {frame}')
                # Confirm if we get a new frame result
                b_get_new_data = frame != frame_get
            except queues.Empty:
                b_get_new_data = False

            if b_get_new_data:
                t_frame_end = time.perf_counter()
                frame_time = t_frame_end - t_frame_start
                frame = frame_get

                if frame and frame % 10 == 0:
                    self.logger.info(f'Predict {subframe} subframe In frame {frame} '
                                     f'by {frame_time} s ({subframe / frame_time} fps)')

                self.save_result_to_file(self.results_save_dir, self.all_predict_result)
                self.logger.debug(f'Save last frame results to file')

                # initial new frame predict result
                self.all_predict_result = {frame: {}}
                subframe = 0
                frame_time = 0

                t_frame_start = time.perf_counter()

                # set new frame track result as new base to predictor
                self.current_predict_result = self.predictor.set_new_base(self.current_track_result)
                self.logger.debug(f'Set Predict base and set base as predict result')

            elif frame:
                self.logger.debug(f'Start subframe predict @ frame {frame}')
                subframe_start_time = time.perf_counter()

                # get current time predict result
                self.current_predict_result = self.predictor.get_predicted_position(subframe_start_time)
                if isinstance(self.current_predict_result, torch.Tensor):
                    self.current_predict_result = self.current_predict_result.numpy()
                subframe += 1

                subframe_time_end = time.perf_counter()
                delta_t_predict = subframe_time_end - subframe_start_time
                self.logger.debug(f'Predicted @ frame {frame} - subframe {subframe} by {delta_t_predict} s')

                if self.output_port.size() < self.output_buffer:
                    self.output_port.send((frame, subframe, self.current_predict_result))
                    self.logger.debug(f'Send predict results to next')
                else:
                    self.logger.debug(f'Output port over {self.output_buffer} buffer size')

                if isinstance(self.current_predict_result, np.ndarray):
                    result_each_subframe = {}
                    result_class = defaultdict(dict)
                    result_id = {}
                    valid_position = np.nonzero(self.current_predict_result)
                    target_num = len(valid_position[0]) // 4

                    for i in range(target_num):
                        cls = valid_position[0][i * 4]
                        target_id = valid_position[1][i * 4]
                        x_position = valid_position[2][(i * 4)]
                        y_position = valid_position[2][(i * 4) + 1]
                        x = self.current_predict_result[cls][target_id][x_position]
                        y = self.current_predict_result[cls][target_id][y_position]
                        result_id[target_id] = ((x, y, 0, 0), 1.0)
                        result_class[cls][target_id] = ((x, y, 0, 0), 1.0)
                        self.logger.debug(f'Predicted class:{cls}, id:{target_id}')
                        # print(i,':',cls,':',result_id)

                    t_subframe_end = time.perf_counter()
                    sub_fps = 1 / (t_subframe_end - t_subframe_start)
                    result_each_subframe[subframe] = (result_class, sub_fps)
                    self.all_predict_result[frame].update(result_each_subframe)
                    self.logger.debug(f'Update subframe result to store dict')

    def run_end(self) -> None:
        super(PathPredictProcess, self).run_end()

        self.logger.info('-'*5 + 'Predict Finished' + '-'*5)

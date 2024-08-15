import multiprocessing
import shutil
import os
import time
import numpy

from . import ConsumerProcess
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType, \
    E_PipelineSharedDataName
from src.multiprocess_pipeline.shared_structure import dict_OutputPortDataType
from src.multiprocess_pipeline.shared_structure import Struc_SharedData, Struc_ConsumerOutputPort
from src.multiprocess_pipeline.workers.matchor import factory_matchor
from src.multiprocess_pipeline.workers.postprocess.utils.write_result import convert_numpy_to_dict
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr


class MultiCameraIdMatchProcess(ConsumerProcess):
    prefix = 'Argos-SubProcess-Global_ID_Match_'
    dir_name = 'id_match'
    log_name = 'ID_Match_Log'
    save_type = [wr.E_text_result_type.raw]
    b_save_in_index = True

    output_type = E_SharedSaveType.Queue
    output_data_type = E_OutputPortDataType.CameraTrack
    output_shape = (1,)

    def __init__(self,
                 matchor_name: str,
                 *args,
                 max_range=10000000,
                 threshold=1,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.port_dict: dict = self.data_hub.dict_consumer_port
        self.matchor_name = matchor_name
        self.max_range = max_range
        self.threshold = threshold

        save_dir = self.results_save_dir[self.pipeline_name]
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        match_result_dir_dict = {}
        for i, l in self.port_dict.items():
            if i != self.pipeline_name:
                match_result_dir_dict[i] = self.making_dir(save_dir, str(i))
        self.results_save_dir = match_result_dir_dict

        self.matchor = None

        self.all_camera_transform = []
        self.all_camera_timestamp = []
        self.match_times = 0

        self_shared_data = self.data_hub.dict_shared_data[self.pipeline_name]
        self.b_read_together = \
            E_PipelineSharedDataName.CameraTransform.name in self_shared_data.keys() \
            and E_PipelineSharedDataName.TransformTimestamp.name in self_shared_data.keys()

    def run_begin(self) -> None:
        super(MultiCameraIdMatchProcess, self).run_begin()

        self.logger.info(f'Get all camera intrinsic parameters')
        intrinsic_parameters_dict = {}
        for pipeline_name, pipeline_shared in self.data_hub.dict_shared_data.items():
            if E_PipelineSharedDataName.CamIntrinsicPara.name in pipeline_shared.keys():
                intrinsic_parameters_dict[pipeline_name] = \
                    pipeline_shared[E_PipelineSharedDataName.CamIntrinsicPara.name].get()
        self.logger.info(f'All camera intrinsic parameters as following:')
        for k, v in intrinsic_parameters_dict.items():
            self.logger.info(f'\n'
                             f'{k}:'
                             f'\n'
                             f'{v}')

        self.logger.info(f'Creating matchor {self.matchor_name}')
        self.matchor = factory_matchor[self.matchor_name](
            intrinsic_parameters_dict,
            self.max_range,
            self.threshold
        )

    def _get_all_camera_transform(self) -> (list, list):
        all_transform = []
        all_timestamp = []

        if self.b_read_together:
            self_shared_data = self.data_hub.dict_shared_data[self.pipeline_name]
            loop_length = min(self_shared_data[E_PipelineSharedDataName.CameraTransform.name].size(),
                              self_shared_data[E_PipelineSharedDataName.TransformTimestamp.name].size())
            for i in range(loop_length):
                dict_transform_data = self_shared_data[E_PipelineSharedDataName.CameraTransform.name].get()
                dict_timestamp = self_shared_data[E_PipelineSharedDataName.TransformTimestamp.name].get()
                if dict_timestamp and dict_transform_data:
                    all_transform.append(dict_transform_data)
                    all_timestamp.append(dict_timestamp)
            all_transform.reverse()
            all_timestamp.reverse()

        else:
            dict_transform_data = {}
            dict_timestamp = {}
            for pipeline in self.port_dict.keys():
                try:
                    data_transform: Struc_SharedData = self.data_hub.dict_shared_data[pipeline][
                        E_PipelineSharedDataName.CameraTransform.name]
                    timestamp_trans: Struc_SharedData = self.data_hub.dict_shared_data[pipeline][
                        E_PipelineSharedDataName.TransformTimestamp.name]
                except KeyError:
                    continue

                dict_transform_data[pipeline] = data_transform
                dict_timestamp[pipeline] = timestamp_trans

            if dict_transform_data and dict_timestamp:
                all_transform.append(dict_transform_data)
                all_timestamp.append(dict_timestamp)

        return all_transform, all_timestamp

    def read_camera_transform_and_timestamp(self, pipeline: str, i: int) -> (numpy.ndarray, int):
        cur_trans = None
        cur_timestamp = None

        if self.all_camera_transform and self.all_camera_timestamp:
            if self.b_read_together:
                cur_trans = self.all_camera_transform[i][pipeline]
                cur_timestamp = self.all_camera_timestamp[i][pipeline]
            else:
                data_transform = self.all_camera_transform[0][pipeline]
                timestamp_trans = self.all_camera_timestamp[0][pipeline]

                cur_trans = data_transform.get()
                cur_timestamp = timestamp_trans.get()

        return cur_trans, cur_timestamp

    def compare_timestamp_get_transform(self, pipeline: str, timestamp_image: int) -> numpy.ndarray:
        last_trans = None
        last_d_timestamp = 999999999999
        loop_length: int

        if self.all_camera_transform and self.all_camera_timestamp:
            if self.b_read_together:
                loop_length = len(self.all_camera_transform)
            else:
                loop_length = self.all_camera_transform[0][pipeline].size()

            for i in range(loop_length):
                cur_trans, cur_timestamp = self.read_camera_transform_and_timestamp(pipeline, i)
                d_timestamp = cur_timestamp - timestamp_image
                if d_timestamp * last_d_timestamp <= 0:
                    if not isinstance(last_trans, numpy.ndarray):
                        last_trans = cur_trans
                        return cur_trans if abs(d_timestamp) > abs(last_d_timestamp) else last_trans
                else:
                    last_trans = cur_trans
                    last_d_timestamp = d_timestamp

        return last_trans

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start globals matching')
        match_result = numpy.empty((2, 2, 2))

        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        while hub_b_loading.value:
            self.all_camera_transform, self.all_camera_timestamp = self._get_all_camera_transform()

            if self.all_camera_transform and self.all_camera_timestamp:
                self.logger.debug('Get camera transform data')

                frame = 0
                subframe = 0
                global_position_last = numpy.empty((1,))
                for pipeline_name, each_pass in self.port_dict.items():
                    t_match_each_start = time.perf_counter()
                    if pipeline_name == self.pipeline_name:
                        continue

                    final_result_port: Struc_ConsumerOutputPort
                    final_result_port = each_pass[-1]

                    if final_result_port.data_type != E_OutputPortDataType.CameraTrack:
                        raise TypeError(f'Pipeline {pipeline_name} last output data type not fit')

                    pipeline_shared_data = self.data_hub.dict_shared_data[pipeline_name]

                    timestamp_image = 0
                    objects_result_content = None
                    objects_result = None
                    b_read_result = True

                    try:
                        timestamp_image = pipeline_shared_data[E_PipelineSharedDataName.ImageTimestamp.name].get()
                        objects_result = final_result_port.read()
                    except multiprocessing.queues.Empty:
                        b_read_result = False
                    if not (timestamp_image and objects_result):
                        b_read_result = False

                    if b_read_result:
                        self.logger.debug(f'Read objects result form camera {pipeline_name} '
                                          f'@ image timestamp {timestamp_image}')
                        frame = objects_result[0]
                        subframe = objects_result[1]
                        objects_result_content = objects_result[2]

                        if not isinstance(objects_result_content,
                                          dict_OutputPortDataType[E_OutputPortDataType.CameraTrack.name][2]):
                            self.logger.debug(f'Read none data from camera {pipeline_name}')
                            break

                        camera_transform = self.compare_timestamp_get_transform(pipeline_name, timestamp_image)
                        if isinstance(camera_transform, numpy.ndarray):
                            self.matchor.camera_transform_dict[pipeline_name] = camera_transform
                            self.logger.debug(f'Set matchor camera transform from camera {pipeline_name}')
                        else:
                            self.logger.debug(f'Get transform from camera {pipeline_name} fail')
                            break

                        if pipeline_name == list(self.port_dict.keys())[0]:
                            self.matchor.baseline_camera_transform = camera_transform
                            self.matchor.baseline_result = objects_result_content
                            match_result = objects_result_content
                            global_position_last = numpy.asarray(objects_result_content)
                            global_position_last.fill(0)
                            self.logger.debug(f'Set matchor baseline object result')

                        else:
                            global_position_last: numpy.ndarray
                            match_result, global_position_current = \
                                self.matchor.get_match_result(pipeline_name, objects_result_content)
                            mask_c = numpy.where(global_position_current[:, :, 3] == 1, 1, 0)
                            mask_l = numpy.where(global_position_last[:, :, 3] == 1, 1, 0)
                            new = mask_c - (mask_c * mask_l)
                            n_new = numpy.nonzero(new)
                            global_position_last[n_new] = mask_c[n_new]

                    else:
                        self.logger.debug(f'No camera {pipeline_name} data')
                        if pipeline_name == list(self.port_dict.keys())[0]:
                            break
                        else:
                            continue

                    t_match_each_end = time.perf_counter()
                    fps = 1 / (t_match_each_end - t_match_each_start)
                    result_frame = convert_numpy_to_dict(match_result, frame, subframe, fps)

                    save_dir = self.results_save_dir[pipeline_name]
                    self.save_result_to_file(save_dir, result_frame)

                self.match_times += 1

                if self.output_port.size() < self.output_buffer:
                    self.logger.debug(f'Send data to next')
                    self.output_port.send((frame, subframe, global_position_last))

            else:
                self.logger.debug(f'No camera transform data')

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Multi Camera Match Finished' + '-' * 5)
        super(MultiCameraIdMatchProcess, self).run_end()

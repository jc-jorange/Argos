import multiprocessing
import shutil
import os
import time
import numpy

from lib.multiprocess_pipeline.process import ConsumerProcess
from lib.multiprocess_pipeline.SharedMemory import Struc_ConsumerOutputPort
from lib.multiprocess_pipeline.workers.matchor import factory_matchor
from lib.multiprocess_pipeline.workers.postprocess.utils.write_result import convert_numpy_to_dict
from lib.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from lib.multiprocess_pipeline.process.SharedDataName import E_PipelineSharedDataName


class MultiCameraIdMatchProcess(ConsumerProcess):
    prefix = 'Argus-SubProcess-Global_ID_Match_'
    dir_name = 'id_match'
    log_name = 'ID_Match_Log'
    save_type = [wr.E_text_result_type.raw]
    b_save_in_index = True

    def __init__(self,
                 matchor_name: str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.port_dict = self.data_hub.dict_consumer_port

        save_dir = self.results_save_dir[self.pipeline_name]
        shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        match_result_dir_dict = {}
        for i, l in self.port_dict.items():
            if i != self.pipeline_name:
                match_result_dir_dict[i] = self.making_dir(save_dir, str(i))
        self.results_save_dir = match_result_dir_dict

        intrinsic_parameters_dict = {}
        for pipeline_name, pipeline_shared in self.data_hub.dict_shared_data.items():
            if E_PipelineSharedDataName.CamIntrinsicPara.name in pipeline_shared.keys():
                intrinsic_parameters_dict[pipeline_name] = \
                    pipeline_shared[E_PipelineSharedDataName.CamIntrinsicPara.name].get()
        self.matchor = factory_matchor[matchor_name](intrinsic_parameters_dict)

        self.all_camera_transform = []
        self.all_camera_timestamp = []

    def _get_all_camera_transform(self) -> (list, list):
        all_transform = []
        all_timestamp = []
        self_shared_data = self.data_hub.dict_shared_data[self.pipeline_name]
        if E_PipelineSharedDataName.CameraTransform.name in self_shared_data.keys() \
                and E_PipelineSharedDataName.TransformTimestamp.name in self_shared_data.keys():
            while self_shared_data[E_PipelineSharedDataName.CameraTransform.name].size()[0] and \
                    self_shared_data[E_PipelineSharedDataName.TransformTimestamp.name].size()[0]:
                dict_transform_data = self_shared_data[E_PipelineSharedDataName.CameraTransform.name].get()
                dict_timestamp = self_shared_data[E_PipelineSharedDataName.TransformTimestamp.name].get()
                if dict_timestamp and dict_transform_data:
                    all_transform.append(dict_transform_data)
                    all_timestamp.append(dict_timestamp)
            all_transform.reverse()
            all_timestamp.reverse()
            return all_transform, all_timestamp

    def compare_timestamp_get_transform(self, pipeline: str, timestamp_image: int) -> numpy.ndarray:
        last_trans = None
        last_d_timestamp = 999999999999

        if self.all_camera_timestamp and self.all_camera_transform:
            for dict_trans, dict_timestamp in zip(self.all_camera_transform, self.all_camera_timestamp):
                cur_trans = dict_trans[pipeline]
                cur_timestamp = dict_timestamp[pipeline]
                d_timestamp = cur_timestamp - timestamp_image
                if d_timestamp*last_d_timestamp <= 0:
                    if not isinstance(last_trans, numpy.ndarray):
                        last_trans = cur_trans
                    if abs(d_timestamp) > abs(last_d_timestamp):
                        return cur_trans
                    else:
                        return last_trans
                else:
                    last_trans = cur_trans
                    last_d_timestamp = d_timestamp
            return last_trans

        else:
            data_transform = self.data_hub.dict_shared_data[pipeline][E_PipelineSharedDataName.CameraTransform.name]
            timestamp_trans = self.data_hub.dict_shared_data[pipeline][E_PipelineSharedDataName.TransformTimestamp.name]
            while data_transform.size():
                cur_trans = data_transform.get()
                cur_timestamp = timestamp_trans.get()
                d_timestamp = cur_timestamp - timestamp_image
                if d_timestamp*last_d_timestamp <= 0:
                    data_transform.clear()
                    timestamp_trans.clear()
                    return cur_trans if abs(d_timestamp) > abs(last_d_timestamp) else last_trans
                else:
                    last_trans = cur_trans
                    last_d_timestamp = d_timestamp

            # return last_trans

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start globals matching')
        result = numpy.empty((2, 2, 2))

        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        while hub_b_loading.value:
            self.all_camera_transform, self.all_camera_timestamp = self._get_all_camera_transform()
            t1 = time.perf_counter()
            match_times = 0

            if self.all_camera_transform and self.all_camera_timestamp:
                for pipeline_name, each_pass in self.port_dict.items():
                    if pipeline_name == self.pipeline_name:
                        continue
                    final_result_port: Struc_ConsumerOutputPort = each_pass[-1]
                    match_times += 1

                    if pipeline_name == list(self.port_dict.keys())[0]:
                        try:
                            timestamp_image = \
                                self.data_hub.dict_shared_data[pipeline_name][E_PipelineSharedDataName.ImageTimestamp.name]\
                                    .get()
                            predict_result = final_result_port.read()
                            frame = predict_result[0]
                            subframe = 0
                            result = predict_result[2]
                            self.matchor.baseline_result = result
                        except multiprocessing.queues.Empty:
                            break

                        if timestamp_image:
                            self.matchor.baseline_camera_transform = \
                                self.compare_timestamp_get_transform(pipeline_name, timestamp_image)
                            self.matchor.camera_transform_dict[pipeline_name] = self.matchor.baseline_camera_transform
                            self.matchor.baseline_result_in_camera = self.matchor.get_baseline_result()
                            if not isinstance(self.matchor.baseline_camera_transform, numpy.ndarray):
                                break
                        else:
                            break

                    else:
                        if not isinstance(self.matchor.baseline_result, numpy.ndarray):
                            continue
                        try:
                            timestamp_image = \
                                self.data_hub.dict_shared_data[pipeline_name][E_PipelineSharedDataName.ImageTimestamp.name]\
                                    .get()
                            if timestamp_image:
                                self.matchor.camera_transform_dict[pipeline_name] = \
                                    self.compare_timestamp_get_transform(pipeline_name, timestamp_image)
                                each_result = final_result_port.read()
                                frame = each_result[0]
                                subframe = each_result[1]
                                each_result = each_result[2]
                                if isinstance(each_result, numpy.ndarray) and \
                                        isinstance(self.matchor.camera_transform_dict[pipeline_name], numpy.ndarray):
                                    result = self.matchor.get_match_result(pipeline_name, each_result)
                        except multiprocessing.queues.Empty:
                            continue

                    t2 = time.perf_counter()
                    fps = match_times / (t2 - t1)
                    result_frame = convert_numpy_to_dict(result, frame, subframe, fps)

                    save_dir = self.results_save_dir[pipeline_name]

                    self.save_result_to_file(save_dir, result_frame)

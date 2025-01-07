import cv2
import numpy
from enum import Enum, unique
import torch

from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType,\
    E_PipelineSharedDataName, E_SharedDataFormat
from . import ConsumerProcess
from src.multiprocess_pipeline.workers.tracker.multitracker import MCJDETracker
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from src.multiprocess_pipeline.workers.tracker.utils.timer import Timer
from src.multiprocess_pipeline.workers.postprocess.utils.write_result import plot_tracks


@unique
class ETrackInfo(Enum):
    Result = 1
    Frame_Index = 2
    Fps = 3


class TrackerProcess(ConsumerProcess):
    prefix = 'Argos-SubProcess-Tracker_'
    dir_name = 'track'
    log_name = 'Track_Log'
    save_type = [wr.E_text_result_type.raw, wr.E_text_result_type.mot]

    output_type = E_SharedSaveType.Queue
    output_data_type = E_OutputPortDataType.CameraTrack
    output_shape = (1,)

    shared_data = {
        E_PipelineSharedDataName.TrackedImageInfo.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },  # [frame_id, timestamp, shape_x, shape_y]
    }

    def __init__(self,
                 arch: str,
                 load_model: str,
                 conf_thres: float,
                 track_buffer: int,
                 *args,
                 min_box_area=10,
                 show_image=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.data_hub.array_schedule_gpu[0] = self.data_hub.array_schedule_gpu[0] + 1
        self.gpu_schedule_index = self.data_hub.array_schedule_gpu[0]

        self.tracker = None
        self.info_data = None
        self.all_frame_results = {}
        self.current_track_result = None

        self.arch = arch
        self.load_model = load_model
        self.conf_thres = conf_thres
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.bshow_image = show_image

        self.timer_loop = Timer()
        self.timer_track = Timer()

        self.frame_id = 0
        self.fps_neuralnetwork_avg = 0
        self.fps_loop_avg = 0
        self.fps_neuralnetwork_current = 0
        self.fps_loop_current = 0

    def run_begin(self) -> None:
        super(TrackerProcess, self).run_begin()

        self.logger.info('Creating tracker')
        self.tracker = MCJDETracker(self.opt,
                                    self.arch,
                                    self.load_model,
                                    self.conf_thres,
                                    self.track_buffer,
                                    self.pipeline_index,
                                    24)
        self.info_data = self.tracker.model.info_data

        self.current_track_result = numpy.zeros([self.info_data.classes_max_num, self.info_data.objects_max_num*2, 4])

    def run_action(self) -> None:
        super(TrackerProcess, self).run_action()
        self.logger.info('Start tracking')

        hub_image = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.Image.name]
        hub_tracked_info = self.data_hub.dict_shared_data[self.pipeline_name][
            E_PipelineSharedDataName.TrackedImageInfo.name]

        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        origin_shape = (0, 0, 0)
        schedule_index_next = 0

        while hub_b_loading.value:
            # loop timer start record
            self.timer_loop.tic()

            image_data = hub_image.get()
            if image_data is None:
                continue

            if self.opt.GPU_load_sequence:
                schedule_index_next = self.data_hub.array_schedule_gpu[1]
                if schedule_index_next != self.gpu_schedule_index:
                    continue

            timestamp = image_data[0]
            input_frame_id = image_data[1]
            img_origin_shape = image_data[2]

            img_0 = image_data[-2]
            img = image_data[-1]

            self.logger.debug(f'delta time as image track get: {self.get_current_timestamp() - timestamp}')

            hub_tracked_info.set([input_frame_id, timestamp, img_origin_shape[0], img_origin_shape[1]])

            if not (isinstance(img, numpy.ndarray) or isinstance(img, torch.Tensor)):
                self.logger.debug(f'Not valid image data')
                continue

            self.logger.debug(f'Tracking Image @ frame {input_frame_id}')

            if self.frame_id == 0:
                origin_shape = img_origin_shape

            # update frame id
            self.frame_id += 1

            # --- run tracking
            # ----- track updates of each frame
            self.timer_track.tic()

            online_targets_dict = self.tracker.update_tracking(img, origin_shape)
            del img

            self.timer_track.toc()
            self.logger.debug(f'Tracking time: {self.timer_track.diff} s')
            # -----

            if self.opt.GPU_load_sequence:
                schedule_index_next += 1
                total_index = self.data_hub.array_schedule_gpu[0]
                self.data_hub.array_schedule_gpu[1] = schedule_index_next if schedule_index_next <= total_index else 0

            result_per_subframe = {}
            total_track_count = 0
            self.current_track_result.fill(0)
            for cls_id in range(self.info_data.classes_max_num):  # process each class id
                online_targets = online_targets_dict[cls_id]
                result_per_subframe[cls_id] = {}
                for track in online_targets:
                    tlwh = track.tlwh
                    xywh = [0, 0, 0, 0]
                    xywh[0] = tlwh[0] + tlwh[2] / 2
                    xywh[1] = tlwh[1] + tlwh[3] / 2
                    xywh[2] = tlwh[2]
                    xywh[3] = tlwh[3]
                    t_id = track.track_id
                    score = track.score
                    if tlwh[2] * tlwh[3] > self.min_box_area:  # and not vertical:
                        result_per_subframe[cls_id][t_id] = (tlwh, score)
                        self.current_track_result[cls_id, t_id] = xywh
                        total_track_count += 1
                        self.logger.debug(f'Tracked class:{cls_id}, id:{t_id}')
            self.all_frame_results[input_frame_id] = {0: (result_per_subframe, self.fps_loop_avg)}
            self.logger.debug(f'Tracked {total_track_count} objects')

            if self.bshow_image and self.opt.allow_show_image:
                show_img = plot_tracks(img_0, self.all_frame_results[input_frame_id], input_frame_id)
                cv2.imshow('Track ' + self.pipeline_name, show_img)
                cv2.waitKey(1)

            self.output_port.send((timestamp, input_frame_id, 0, self.current_track_result))
            self.logger.debug(f'Send track results to next at timestamp {self.get_current_timestamp()}')
            self.logger.debug(f'delta time as image track send: {self.get_current_timestamp() - timestamp}')

            # loop timer end record
            self.timer_loop.toc()

            self.fps_loop_avg = self.frame_id / max(1e-5, self.timer_loop.total_time)
            self.fps_loop_current = 1.0 / max(1e-5, self.timer_loop.diff)

            self.fps_neuralnetwork_avg = self.frame_id / max(1e-5, self.timer_track.total_time)
            self.fps_neuralnetwork_current = 1.0 / max(1e-5, self.timer_track.diff)

            b_save = self.save_result_to_file(self.results_save_dir, self.all_frame_results)
            if b_save:
                self.logger.debug(f'Save track results to file {self.results_save_dir}')
            else:
                self.logger.info(f'SAVE RESULTS FAIL!!!!!')
            self.all_frame_results = {}

            if self.frame_id % 10 == 0 and self.frame_id != 0:
                self.logger.info(
                    f'Processing frame {self.frame_id}: '
                    f'loop average fps: {self.fps_loop_avg:.2f}, '
                    f'loop current fps: {self.fps_loop_current:.2f}; '
                    f'track average fps: {self.fps_neuralnetwork_avg:.2f}, '
                    f'track current fps: {self.fps_neuralnetwork_current:.2f}'
                )

            self.logger.debug(
                f'Processing frame {self.frame_id}: {self.fps_neuralnetwork_current:.2f} track current fps,'
                f' {1.0 / self.fps_neuralnetwork_current} s'
            )
            self.logger.debug(
                f'Processing frame {self.frame_id}: {self.fps_loop_current:.2f} '
                f'loop current fps, {1.0 / self.fps_loop_current} s'
            )

    def run_end(self) -> None:
        self.logger.info(f'Final loop time {self.timer_loop.total_time}')
        self.logger.info(f'Final loop FPS {self.fps_loop_avg}')
        self.logger.info(f'Final inference time {self.timer_track.total_time}')
        self.logger.info(f'Final inference FPS {self.fps_neuralnetwork_avg}')

        super(TrackerProcess, self).run_end()

        self.logger.info('-'*5 + 'Tracker Finished' + '-'*5)

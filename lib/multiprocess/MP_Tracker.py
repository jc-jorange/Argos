import ctypes
import time

import numpy
from enum import Enum, unique
import torch

from .SharedMemory import SharedContainer, EQueueType, EResultType
from ..multiprocess import BaseProcess
from lib.tracker.multitracker import MCJDETracker
from ..postprocess.utils import write_result as wr
from lib.tracker.utils.timer import Timer


@unique
class ETrackInfo(Enum):
    Result = 1
    Frame_Index = 2
    Fps = 3


class TrackerProcess(BaseProcess):
    prefix = 'Argus-SubProcess-Tracker_'
    dir_name = 'track'
    log_name = 'Track_Log'
    save_type = [wr.E_text_result_type.raw, wr.E_text_result_type.mot]

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.tracker = None
        self.info_data = None
        self.all_frame_results = wr.S_default_result
        self.current_track_result = None

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
        self.tracker = MCJDETracker(self.opt, 24)
        self.info_data = self.tracker.model.info_data

        self.current_track_result = numpy.zeros([self.info_data.classes_max_num, self.info_data.objects_max_num, 4])

    def run_action(self) -> None:
        super(TrackerProcess, self).run_action()
        self.logger.info('Start tracking')

        input_to_track_queue = self.shared_container.queue_dict[EQueueType.LoadResultSend]
        track_to_predict_queue = self.shared_container.queue_dict[EQueueType.TrackerResultSend]

        origin_shape = (0, 0, 0)

        while self.shared_container.b_input_loading.value:
            # loop timer start record
            self.timer_loop.tic()

            if not self.opt.realtime:
                try:
                    input_frame_id, img, origin_shape = input_to_track_queue.get(block=False)
                except:
                    continue
            else:
                input_frame_id = self.shared_container.input_frame_id.value
                img = self.shared_container.resized_tensor

                if self.frame_id == 1:
                    origin_shape = self.shared_container.get_origin_shape()

            # update frame id
            self.frame_id += 1

            # --- run tracking
            # ----- track updates of each frame
            if self.frame_id > 1:
                self.timer_track.tic()

            online_targets_dict = self.tracker.update_tracking(img, origin_shape)

            if self.frame_id > 1:
                self.timer_track.toc()
            # -----

            result_per_subframe = {}
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
                    if tlwh[2] * tlwh[3] > self.opt.min_box_area:  # and not vertical:
                        result_per_subframe[cls_id][t_id] = (tlwh, score)
                        self.current_track_result[cls_id, t_id] = xywh
            self.all_frame_results[input_frame_id] = {0: (result_per_subframe, self.fps_loop_avg)}

            # result_numpy = numpy.frombuffer(self.shared_container.result_dict[EResultType.TrackResult], dtype=ctypes.c_float)

            track_to_predict_queue.put((self.frame_id, self.current_track_result))

            # loop timer end record
            self.timer_loop.toc()

            del img

            self.fps_loop_avg = self.frame_id / max(1e-5, self.timer_loop.total_time)
            self.fps_loop_current = 1.0 / max(1e-5, self.timer_loop.diff)

            self.fps_neuralnetwork_avg = self.frame_id / max(1e-5, self.timer_track.total_time)
            self.fps_neuralnetwork_current = 1.0 / max(1e-5, self.timer_track.diff)

            if self.frame_id % 10 == 0 and self.frame_id != 0:
                self.save_result_to_file(self.main_output_dir, self.all_frame_results)
                self.all_frame_results = wr.S_default_result

                self.logger.info(
                    f'Processing frame {self.frame_id}: '
                    f'loop average fps: {self.fps_loop_avg:.2f}, '
                    f'loop current fps: {self.fps_loop_current:.2f}; '
                    f'track average fps: {self.fps_neuralnetwork_avg:.2f}, '
                    f'track current fps: {self.fps_neuralnetwork_current:.2f}'
                )

            self.logger.debug(
                'Processing frame {}: {:.2f} track current fps, {} s'.format(
                    self.frame_id,
                    self.fps_neuralnetwork_current,
                    1.0 / self.fps_neuralnetwork_current
                )
            )
            self.logger.debug(
                'Processing frame {}: {:.2f} loop current fps, {} s'.format(
                    self.frame_id,
                    self.fps_loop_current,
                    1.0 / self.fps_loop_current
                )
            )

        self.save_result_to_file(self.main_output_dir, self.all_frame_results)
        del self.all_frame_results

        while track_to_predict_queue.qsize() > 0:
            track_to_predict_queue.get()

    def run_end(self) -> None:
        super().run_end()
        self.logger.info(f'Final loop time {self.timer_loop.total_time}')
        self.logger.info(f'Final loop FPS {self.fps_loop_avg}')
        self.logger.info(f'Final inference time {self.timer_track.total_time}')
        self.logger.info(f'Final inference FPS {self.fps_neuralnetwork_avg}')

        self.logger.info('-'*5 + 'Tracker Finished' + '-'*5)

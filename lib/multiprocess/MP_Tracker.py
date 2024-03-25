import numpy
from enum import Enum, unique

from ..multiprocess import BaseProcess, EMultiprocess
from lib.tracker.multitracker import MCJDETracker
from lib.tracker.utils.utils import *
from ..postprocess.utils import write_result as wr
from lib.tracker.utils.timer import Timer
from Main import EQueueType


@unique
class ETrackInfo(Enum):
    Result = 1
    Frame_Index = 2
    Fps = 3


class TrackerProcess(BaseProcess):
    prefix = 'Argus-SubProcess-Tracker_'
    dir_name = 'track'

    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.tracker = None
        self.info_data = None
        self.all_track_results = wr.Total_Result_Format
        self.current_track_result = None

        self.timer_loop = Timer()
        self.timer_track = Timer()

        self.result_file_name = os.path.join(self.main_output_dir, 'text_result.txt')

        self.frame_id = 0
        self.fps_neuralnetwork_avg = 0
        self.fps_loop_avg = 0
        self.fps_neuralnetwork_current = 0
        self.fps_loop_current = 0

    def run_begin(self) -> None:
        super(TrackerProcess, self).run_begin()
        self.set_logger_file_handler(self.name + '_Tracker_Log', self.main_output_dir)
        self.logger.info("This is the Tracker Process No.{:d}".format(self.idx))

        self.logger.info('Creating tracker')
        self.tracker = MCJDETracker(self.opt, 24)
        self.info_data = self.tracker.model.info_data

        self.all_track_results = {cls_id: [] for cls_id in range(self.info_data.classes_max_num)}
        self.all_track_results = wr.Total_Result_Format
        self.current_track_result = numpy.zeros([self.info_data.classes_max_num, self.info_data.objects_max_num, 4])

    def run_action(self) -> None:
        super(TrackerProcess, self).run_action()
        self.logger.info('Start tracking')

        input_queue = self.container_queue[EQueueType.InputToTracker][self.idx]
        track_queue = self.container_queue[EQueueType.TrackerToPredict][self.idx]
        while not self.end_run_flag.value:
            # loop timer start record
            self.timer_loop.tic()

            input_data = input_queue.get()

            if not input_data:
                break

            else:
                path, img, img0 = input_data

                # --- run tracking
                blob = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)

                # ----- track updates of each frame
                self.timer_track.tic()

                online_targets_dict = self.tracker.update_tracking(blob, img0)

                self.timer_track.toc()
                # -----

                for cls_id in range(self.info_data.classes_max_num):  # process each class id
                    online_targets = online_targets_dict[cls_id]
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
                            result_per_frame = wr.Track_Result_Format
                            result_per_frame[cls_id][t_id] = (tlwh, score)
                            result_and_fps = [result_per_frame, self.fps_loop_avg]
                            self.all_track_results[self.frame_id + 1][0] = result_and_fps
                            self.current_track_result[cls_id, t_id] = xywh

                track_queue.put(self.current_track_result)

                # update frame id
                self.frame_id += 1

                self.fps_loop_avg = self.frame_id / max(1e-5, self.timer_loop.total_time)
                self.fps_loop_current = 1.0 / max(1e-5, self.timer_loop.diff)

                self.fps_neuralnetwork_avg = self.frame_id / max(1e-5, self.timer_track.total_time)
                self.fps_neuralnetwork_current = 1.0 / max(1e-5, self.timer_track.diff)

                if self.frame_id % 10 == 0 and self.frame_id != 0:
                    self.logger.info(
                        f'Processing frame {self.frame_id}: '
                        f'loop average fps: {self.fps_loop_avg:.2f}, '
                        f'loop current fps: {self.fps_loop_current:.2f}; '
                        f'track average fps: {self.fps_neuralnetwork_avg:.2f}, '
                        f'track current fps: {self.fps_neuralnetwork_current:.2f}'
                    )
                self.logger.debug(
                    'Processing frame {}: {:.2f} track current fps, {} s'.format(
                        self.frame_id, self.fps_neuralnetwork_current, 1.0 / self.fps_neuralnetwork_current
                    )
                )
                self.logger.debug(
                    'Processing frame {}: {:.2f} loop current fps, {} s'.format(
                        self.frame_id, self.fps_loop_current, 1.0 / self.fps_loop_current
                    )
                )

            # loop timer end record
            self.timer_loop.toc()

    def run_end(self) -> None:
        super().run_end()

        self.logger.info('Final loop time {}'.format(self.timer_loop.total_time))
        self.logger.info('Final loop FPS {}'.format(self.fps_loop_avg))
        self.logger.info('Final inference time {}'.format(self.timer_track.total_time))
        self.logger.info('Final inference FPS {}'.format(self.fps_neuralnetwork_avg))

        self.container_result[EMultiprocess.Tracker][self.idx] = self.all_track_results

        self.logger.info('-'*5 + 'Tracker Finished' + '-'*5)

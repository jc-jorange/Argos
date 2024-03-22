import numpy
from collections import defaultdict
from multiprocessing import Process
from enum import Enum, unique

from ..multiprocess import BaseProcess, ESharedDictType, EMultiprocess
from .MP_ImageReceiver import EImageInfo
from lib.tracker.multitracker import MCJDETracker
from lib.tracker.utils.utils import *
from lib.tracker.utils import write_result as wr, visualization as vis
from lib.tracker.utils.timer import Timer
from lib.dataset.__init__ import LoadData


@unique
class ETrackInfo(Enum):
    Result = 1


class TrackerProcess(BaseProcess):
    prefix = 'Argus-SubProcess-Tracker_'

    def __init__(self,
                 *args,
                 frame_rate=24,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        # self.frame_rate = frame_rate

        self.making_process_main_save_dir('camera_')

        self.timer_loop = Timer()
        self.timer_track = Timer()

        self.result_file_name = os.path.join(self.main_output_dir, 'text_result.txt')
        self.frame_dir = None if self.opt.output_format == 'text' \
            else self.making_dir(self.main_output_dir, 'frame')[0]

        self.frame_id = 0
        self.fps_neuralnetwork_avg = 0
        self.fps_loop_avg = 0
        self.fps_neuralnetwork_current = 0
        self.fps_loop_current = 0

    def run_begin(self) -> None:
        super(TrackerProcess, self).run_begin()
        self.data_loader = LoadData(self.idx, self.opt.input_mode, self.opt.input_path[self.idx], self.container_shared_dict[ESharedDictType.Image])

        self.set_logger_file_handler(self.name + '_Tracker_Log', self.main_output_dir)
        self.logger.info("This is the Tracker Process No.{:d}".format(self.idx))

        self.logger.info('Creating tracker')
        self.tracker = MCJDETracker(self.opt, 24)
        self.info_data = self.tracker.model.info_data

        self.results_dict = {cls_id: [] for cls_id in range(self.info_data.classes_max_num)}

    
    def run_action(self) -> None:
        super(TrackerProcess, self).run_action()
        self.logger.info('Start tracking')
        self.timer_loop.tic()
        # torch.cuda.set_per_process_memory_fraction(0.5, 0)
        for path, img, img0 in self.data_loader:
            # loop timer end record
            self.timer_loop.toc()
            # loop timer start record
            self.timer_loop.tic()
            # --- run tracking
            blob = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)

            # ----- track updates of each frame
            if self.frame_id > 0:
                self.timer_track.tic()

            online_targets_dict = self.tracker.update_tracking(blob, img0)

            if self.frame_id > 0:
                self.timer_track.toc()
            # -----

            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)
            current_result = numpy.zeros([self.info_data.classes_max_num, self.info_data.objects_max_num, 4])

            for cls_id in range(self.info_data.classes_max_num):  # process each class id
                online_targets = online_targets_dict[cls_id]
                for track in online_targets:
                    tlwh = track.tlwh
                    xywh = [0,0,0,0]
                    xywh[0] = tlwh[0] + tlwh[2]/2
                    xywh[1] = tlwh[1] + tlwh[3]/2
                    xywh[2] = tlwh[2]
                    xywh[3] = tlwh[3]
                    t_id = track.track_id
                    score = track.score
                    if tlwh[2] * tlwh[3] > self.opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)
                        self.results_dict[cls_id].append((self.frame_id + 1,
                                                 online_tlwhs_dict[cls_id],
                                                 online_ids_dict[cls_id],
                                                 online_scores_dict[cls_id]))
                        current_result[cls_id, t_id] = xywh

            self.container_shared_dict[ESharedDictType.Track][self.idx].set(ETrackInfo.Result, current_result)

            self.container_shared_dict[ESharedDictType.Image][self.idx].set(EImageInfo.Data, img0)
            self.container_shared_dict[ESharedDictType.Image][self.idx].set(EImageInfo.Size, img0.size)

            self.fps_loop_avg = self.frame_id / max(1e-5, self.timer_loop.total_time)
            self.fps_loop_current = 1.0 / max(1e-5, self.timer_loop.diff)

            # post
            # draw track/detection
            if self.frame_id > 0:
                online_im = vis.plot_tracks(image=img0,
                                            tlwhs_dict=online_tlwhs_dict,
                                            obj_ids_dict=online_ids_dict,
                                            num_classes=self.info_data.classes_max_num,
                                            frame_id=self.frame_id,
                                            fps=self.fps_loop_avg)

                if self.opt.show_image:
                    cv2.imshow('online_im', online_im)

                if self.frame_dir:
                    cv2.imwrite(os.path.join(self.frame_dir, '{:05d}.jpg'.format(self.frame_id)), online_im)

            # update frame id
            self.frame_id += 1

            self.fps_neuralnetwork_avg = self.frame_id / max(1e-5, self.timer_track.total_time)
            self.fps_neuralnetwork_current = 1.0 / max(1e-5, self.timer_track.diff)
            if self.frame_id % 10 == 0 and self.frame_id != 0:
                self.logger.info(
                    'Processing frame {}: {:.2f} loop average fps, {:.2f} loop current fps; '
                    '{:.2f} track average fps, {:.2f} track current fps'.format(
                        self.frame_id,
                        self.fps_loop_avg,
                        self.fps_loop_current,
                        self.fps_neuralnetwork_avg,
                        self.fps_neuralnetwork_current,
                    )
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

        self.timer_loop.toc()
        self.logger.info('Final loop time {}'.format(self.timer_loop.total_time))
        self.logger.info('Final loop FPS {}'.format(self.fps_loop_avg))
        self.logger.info('Final inference time {}'.format(self.timer_track.total_time))
        self.logger.info('Final inference FPS {}'.format(self.fps_neuralnetwork_avg))
        # write track/detection results to text
        wr.write_results_to_text(self.result_file_name,
                                 self.results_dict,
                                 'mot',
                                 self.info_data.classes_max_num)
        self.logger.info('Saving result in {}'.format(self.result_file_name))

        # write results to video
        if self.opt.output_format == 'video':
            wr.write_results_to_video(self.main_output_dir,
                                      self.frame_dir,
                                      ('m', 'p', '4', 'v'),
                                      24)
            self.logger.info('Saving result video to {}, format as mp4'.format(self.main_output_dir))

        self.logger.info('-'*5 + 'Tracker Finished' + '-'*5)


import datetime
import time
import numpy

from collections import defaultdict
from multiprocessing import Process

import lib.multiprocess.Shared as Sh
from lib.tracker.multitracker import MCJDETracker
from lib.tracker.utils.utils import *
from lib.tracker.utils import write_result as wr, visualization as vis
from lib.tracker.utils.timer import Timer
from lib.dataset.__init__ import LoadData
from lib.utils.logger import logger


class Tracker_Process(Process):
    def __init__(self,
                 idx: int,
                 opt,
                 shared_image_dict: Sh.SharedDict,
                 frame_rate=24):
        super().__init__()
        self.name = 'Argus-SubProcess-Tracker_' + str(idx)
        self.idx = idx
        self.opt = opt
        self.frame_rate = frame_rate

        self.result_root = os.path.join(opt.save_dir, 'camera_' + str(idx+1))
        mkdir_if_missing(self.result_root)

        self.shared_image_dict = shared_image_dict

        self.data_loader = LoadData(self.idx, opt.input_mode, opt.input_path[idx], self.shared_image_dict)

        self.result_file_name = os.path.join(self.result_root, 'text_result.txt')
        self.frame_dir = None if opt.output_format == 'text' \
            else os.path.join(self.result_root, 'frame')
        if self.frame_dir:
            mkdir_if_missing(self.frame_dir)

        self.fps_neuralnetwork_avg = 0
        self.fps_loop_avg = 0
        self.fps_neuralnetwork_current = 0
        self.fps_loop_current = 0

        self.logger = None

    def run(self):
        opt = self.opt

        self.logger = logger.add_logger(os.getpid())
        logger.add_stream_handler(os.getpid())
        logger.add_file_handler(os.getpid(), self.name, self.result_root)

        logger.set_logger_level(os.getpid(), 'debug' if self.opt.debug else 'info')
        self.logger.info("This is the Tracker Process No.{:d}".format(self.idx))

        self.logger.info('Creating tracker')
        tracker = MCJDETracker(opt, self.frame_rate)
        info_data = tracker.model.info_data

        timer_loop = Timer()
        timer_track = Timer()

        results_dict = {cls_id: [] for cls_id in range(info_data.classes_max_num)}

        logger.logger_dict[os.getpid()].info('Start tracking')
        frame_id = 0  # frame index
        timer_loop.tic()

        # torch.cuda.set_per_process_memory_fraction(0.5, 0)

        for path, img, img0 in self.data_loader:
            # loop timer end record
            timer_loop.toc()
            # loop timer start record
            timer_loop.tic()
            # --- run tracking
            blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

            # ----- track updates of each frame
            if frame_id > 0:
                timer_track.tic()

            online_targets_dict = tracker.update_tracking(blob, img0)

            if frame_id > 0:
                timer_track.toc()
            # -----

            # collect current frame's result
            online_tlwhs_dict = defaultdict(list)
            online_ids_dict = defaultdict(list)
            online_scores_dict = defaultdict(list)
            current_result = numpy.zeros([info_data.classes_max_num, info_data.objects_max_num, 4])

            for cls_id in range(info_data.classes_max_num):  # process each class id
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
                    if tlwh[2] * tlwh[3] > opt.min_box_area:  # and not vertical:
                        online_tlwhs_dict[cls_id].append(tlwh)
                        online_ids_dict[cls_id].append(t_id)
                        online_scores_dict[cls_id].append(score)
                        results_dict[cls_id].append((frame_id + 1,
                                                 online_tlwhs_dict[cls_id],
                                                 online_ids_dict[cls_id],
                                                 online_scores_dict[cls_id]))
                        current_result[cls_id, t_id] = xywh

            self.fps_loop_avg = frame_id / max(1e-5, timer_loop.total_time)
            self.fps_loop_current = 1.0 / max(1e-5, timer_loop.diff)

            # post
            # draw track/detection
            if frame_id > 0:
                online_im = vis.plot_tracks(image=img0,
                                            tlwhs_dict=online_tlwhs_dict,
                                            obj_ids_dict=online_ids_dict,
                                            num_classes=info_data.classes_max_num,
                                            frame_id=frame_id,
                                            fps=self.fps_loop_avg)

                if opt.show_image:
                    cv2.imshow('online_im', online_im)

                if self.frame_dir:
                    cv2.imwrite(os.path.join(self.frame_dir, '{:05d}.jpg'.format(frame_id)), online_im)

            # update frame id
            frame_id += 1

            self.fps_neuralnetwork_avg = frame_id / max(1e-5, timer_track.total_time)
            self.fps_neuralnetwork_current = 1.0 / max(1e-5, timer_track.diff)
            if frame_id % 10 == 0 and frame_id != 0:
                logger.logger_dict[os.getpid()].info(
                    'Processing frame {}: {:.2f} loop average fps, {:.2f} loop current fps; '
                    '{:.2f} track average fps, {:.2f} track current fps'.format(
                        frame_id,
                        self.fps_loop_avg,
                        self.fps_loop_current,
                        self.fps_neuralnetwork_avg,
                        self.fps_neuralnetwork_current,
                    )
                )
            logger.logger_dict[os.getpid()].debug(
                'Processing frame {}: {:.2f} track current fps, {} s'.format(
                    frame_id, self.fps_neuralnetwork_current, 1.0 / self.fps_neuralnetwork_current
                )
            )
            logger.logger_dict[os.getpid()].debug(
                'Processing frame {}: {:.2f} loop current fps, {} s'.format(
                    frame_id, self.fps_loop_current, 1.0 / self.fps_loop_current
                )
            )

        timer_loop.toc()
        self.logger.info('Final loop time {}'.format(timer_loop.total_time))
        self.logger.info('Final loop FPS {}'.format(self.fps_loop_avg))
        self.logger.info('Final inference time {}'.format(timer_track.total_time))
        self.logger.info('Final inference FPS {}'.format(self.fps_neuralnetwork_avg))
        # write track/detection results to text
        wr.write_results_to_text(self.result_file_name,
                                 results_dict,
                                 'mot',
                                 info_data.classes_max_num)
        self.logger.info('Saving result in {}'.format(self.result_file_name))

        # write results to video
        if self.opt.output_format == 'video':
            wr.write_results_to_video(self.result_root,
                                      self.frame_dir,
                                      ('m', 'p', '4', 'v'),
                                      self.frame_rate)
            self.logger.info('Saving result video to {}, format as mp4'.format(self.result_root))

        self.logger.info('-'*5 + 'Tracker Finished' + '-'*5)

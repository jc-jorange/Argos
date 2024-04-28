import os

from lib.multiprocess import PostProcess
from lib.multiprocess.individual_process import E_Indi_Process_Producer, E_Indi_Process_Consumer
from lib.postprocess import BasePost
from lib.postprocess.result_writer import ImageResultWriter, VideoResultWriter
import lib.postprocess.utils.write_result as wr


class IndividualPostProcess(PostProcess):
    prefix = 'Argus-SubProcess-IndiPostProcess_'

    def __init__(self,
                 process_dir: {},
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_list = []
        self.process_dir = process_dir

    def run_begin(self) -> None:
        image_loader_dir = self.process_dir[self.idx][E_Indi_Process_Producer.ImageLoader.name]
        frame_dir = os.path.join(image_loader_dir, self.opt.frame_dir)

        track_dir = self.process_dir[self.idx][E_Indi_Process_Consumer.Tracker.name]
        track_result = os.path.join(track_dir, wr.Dict_text_result_name[wr.E_text_result_type.raw])
        track_image_save_dir = self.making_dir(track_dir, self.opt.frame_dir)
        track_image_writer = ImageResultWriter(frame_dir, track_image_save_dir, track_result)
        self.post_process_list.append(track_image_writer)

        predict_dir = self.process_dir[self.idx][E_Indi_Process_Consumer.Predictor.name]
        predict_result = os.path.join(predict_dir, wr.Dict_text_result_name[wr.E_text_result_type.raw])
        predict_image_save_dir = self.making_dir(predict_dir, self.opt.frame_dir)
        predict_image_writer = ImageResultWriter(frame_dir, predict_image_save_dir, predict_result)
        self.post_process_list.append(predict_image_writer)

        if self.opt.output_format == 'video':
            track_video_writer = VideoResultWriter(track_image_save_dir, track_dir)
            self.post_process_list.append(track_video_writer)

            predict_video_writer = VideoResultWriter(predict_image_save_dir, predict_dir)
            self.post_process_list.append(predict_video_writer)

    def run_action(self) -> None:
        super().run_action()

        each_post: BasePost
        for each_post in self.post_process_list:
            each_post.process()
            self.logger.info('Saving result')

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Saving Finished' + '-' * 5)

import os.path

from ..multiprocess import BaseProcess, EMultiprocess
from lib.postprocess import BasePost
from lib.postprocess.write_result_to_image import TextResultWriter, ImageResultWriter, VideoResultWriter


class IndividualPostProcess(BaseProcess):
    prefix = 'Argus-SubProcess-IndiPostProcess_'

    def __init__(self,
                 process_dir: {},
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_list = []
        self.process_dir = process_dir

        frame_dir = self.container_result[EMultiprocess.ImageReceiver][self.idx]
        track_result = self.container_result[EMultiprocess.Tracker][self.idx]
        predict_result = self.container_result[EMultiprocess.Predictor][self.idx]

        track_dir = process_dir[EMultiprocess.Tracker][self.idx]
        track_text_writer = TextResultWriter(track_dir, 'text_result.txt', track_result, dtype='mot')
        self.post_process_list.append(track_text_writer)

        track_image_save_dir = str(os.path.join(track_dir, self.opt.frame_dir))
        track_image_writer = ImageResultWriter(frame_dir, track_image_save_dir, track_result, self.opt)
        self.post_process_list.append(track_image_writer)

        predict_dir = process_dir[EMultiprocess.Predictor][self.idx]
        predict_text_writer = TextResultWriter(predict_dir, 'text_result.txt', predict_result, dtype='raw')
        self.post_process_list.append(predict_text_writer)

        predict_image_save_dir = str(os.path.join(predict_dir, self.opt.frame_dir))
        predict_image_writer = ImageResultWriter(frame_dir, predict_image_save_dir, predict_result, self.opt)
        self.post_process_list.append(predict_image_writer)

        if self.opt.output_format == 'video':
            track_video_writer = VideoResultWriter(track_image_save_dir, track_dir, self.opt, frame_rate=24)
            self.post_process_list.append(track_video_writer)

            predict_video_writer = VideoResultWriter(predict_image_save_dir, predict_dir, self.opt, frame_rate=24)
            self.post_process_list.append(predict_video_writer)

    def run_action(self) -> None:
        super().run_action()

        each_post: BasePost
        for each_post in self.post_process_list:
            each_post.process()
            self.logger.info('Saving result')

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Saving Finished' + '-' * 5)

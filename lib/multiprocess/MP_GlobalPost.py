import os

from ..multiprocess import BaseProcess, EMultiprocess
from lib.postprocess import BasePost
from lib.postprocess.result_writer import ImageResultWriter, VideoResultWriter
import lib.postprocess.utils.write_result as wr


class GlobalPostProcess(BaseProcess):
    prefix = 'Argus-SubProcess-GlobalPostProcess_'

    def __init__(self,
                 process_dir: {},
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_list = []
        self.process_dir = process_dir

    def run_begin(self) -> None:
        for i_process, each_dir_dict in self.process_dir.items():
            image_loader_dir = each_dir_dict[EMultiprocess.ImageLoader]
            frame_dir = os.path.join(image_loader_dir, self.opt.frame_dir)

            match_dir = each_dir_dict[EMultiprocess.GlobalMatching]
            match_result = os.path.join(match_dir, wr.Dict_text_result_name[wr.E_text_result_type.raw])
            match_image_save_dir = self.making_dir(match_dir, self.opt.frame_dir)
            match_image_writer = ImageResultWriter(frame_dir, match_image_save_dir, match_result, self.opt)
            self.post_process_list.append(match_image_writer)

            if self.opt.output_format == 'video':
                track_video_writer = VideoResultWriter(match_image_save_dir, match_dir, self.opt)
                self.post_process_list.append(track_video_writer)


    def run_action(self) -> None:
        super().run_action()

        each_post: BasePost
        for each_post in self.post_process_list:
            each_post.process()
            self.logger.info('Saving result')

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Saving Finished' + '-' * 5)

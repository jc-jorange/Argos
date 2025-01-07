import os
import time

from .._masterclass import PostProcess
from src.multiprocess_pipeline.process import E_pipeline_branch
from src.multiprocess_pipeline.process.producer import E_Process_Producer
from src.multiprocess_pipeline.workers.postprocess import BasePost
from src.multiprocess_pipeline.workers.postprocess.result_writer import ImageResultWriter, VideoResultWriter
import src.multiprocess_pipeline.workers.postprocess.utils.write_result as wr


class GloResultsVisualizeProcess(PostProcess):
    prefix = 'Argos-SubProcess-GlobalPostProcess_'
    dir_name = 'global_post'
    log_name = 'Global_Post_Log'

    def __init__(self,
                 output_format: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_list = []
        self.output_format = output_format

    def run_begin(self) -> None:
        for process_name, process_result_dir in self.data_hub.dict_process_results_dir[self.pipeline_name] \
                [E_pipeline_branch.consumer.name].items():
            if isinstance(process_result_dir, dict):
                for k, each_dir in process_result_dir.items():
                    each_producer_dict = self.data_hub.dict_process_results_dir[k][E_pipeline_branch.producer.name]
                    each_producer_dict: dict
                    if E_Process_Producer.ImageLoader.name not in each_producer_dict.keys():
                        continue
                    each_frame_dir = self.data_hub.dict_process_results_dir[k] \
                        [E_pipeline_branch.producer.name][E_Process_Producer.ImageLoader.name]
                    each_result = os.path.join(each_dir, wr.Dict_text_result_name[wr.E_text_result_type.raw])
                    each_image_save_dir = self.making_dir(each_dir, wr.Str_image_result_dir_name)
                    each_image_writer = ImageResultWriter(each_frame_dir, each_image_save_dir, each_result)
                    self.post_process_list.append(each_image_writer)

                    if self.output_format == 'video':
                        each_video_writer = VideoResultWriter(each_image_save_dir, each_dir)
                        self.post_process_list.append(each_video_writer)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info(f'Start global post saving')
        each_post: BasePost
        for each_post in self.post_process_list:
            try:
                each_post.process()
                self.logger.info('Saving result')
            except FileNotFoundError:
                self.logger.info(f'No such file or directory: {each_post.save_dir}')
                pass

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Saving Finished' + '-' * 5)

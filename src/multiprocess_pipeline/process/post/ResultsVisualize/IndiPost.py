import os

from ..ResultsVisualize import ResultsVisualizeProcess_Master
from src.multiprocess_pipeline.process.interface import E_pipeline_branch
from src.multiprocess_pipeline.process.producer import E_Process_Producer
from src.multiprocess_pipeline.workers.postprocess import BasePost
from src.multiprocess_pipeline.workers.postprocess.result_writer import ImageResultWriter, VideoResultWriter
import src.multiprocess_pipeline.workers.postprocess.utils.write_result as wr


class IndiResultsVisualizeProcess(ResultsVisualizeProcess_Master):
    prefix = 'Argus-SubProcess-IndiPostProcess_'
    dir_name = 'indi_post'
    log_name = 'Indi_Post_Log'

    def __init__(self,
                 output_format: str,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_list = []
        self.output_format = output_format

    def run_begin(self) -> None:
        frame_dir = \
            self.data_hub.dict_process_results_dir[self.pipeline_name][E_pipeline_branch.producer.name] \
                [E_Process_Producer.ImageLoader.name]

        consumer_image_dir_dict = {}
        for process_name, process_result_dir in self.data_hub.dict_process_results_dir[self.pipeline_name] \
                [E_pipeline_branch.consumer.name].items():
            consumer_result = os.path.join(process_result_dir, wr.Dict_text_result_name[wr.E_text_result_type.raw])
            consumer_image_save_dir = self.making_dir(process_result_dir, wr.Str_image_result_dir_name)
            consumer_image_dir_dict[process_name] = consumer_image_save_dir
            consumer_image_writer = ImageResultWriter(frame_dir, consumer_image_save_dir, consumer_result)
            self.post_process_list.append(consumer_image_writer)

        if self.output_format == 'video':
            for process_name, process_result_dir in self.data_hub.dict_process_results_dir[self.pipeline_name] \
                    [E_pipeline_branch.consumer.name].items():
                consumer_video_writer = VideoResultWriter(consumer_image_dir_dict[process_name], process_result_dir)
                self.post_process_list.append(consumer_video_writer)

    def run_action(self) -> None:
        super().run_action()

        while self.data_hub.dict_bLoadingFlag[self.pipeline_name].value:
            pass

        self.logger.info(f'Start indi post saving')
        each_post: BasePost
        for each_post in self.post_process_list:
            each_post.process()
            self.logger.info('Saving result')

    def run_end(self) -> None:
        self.logger.info('-' * 5 + 'Saving Finished' + '-' * 5)

import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.tracker.utils.utils import mkdir_if_missing
import lib.postprocess.utils.write_result as wr
from lib.multiprocess.SharedMemory import ProducerBucket, ConsumerOutputPort


class BaseProcess(Process):
    prefix = ''
    dir_name = ''
    log_name = ''
    save_type = []

    def __init__(self,
                 shared_container: ProducerBucket,
                 idx: int,
                 opt: opts,
                 ) -> None:
        super(BaseProcess, self).__init__()
        self.check_save_type(self.save_type)

        self.shared_container = shared_container
        self.idx = idx
        self.opt = opt

        self.all_frame_results = wr.S_default_result

        self.name = self.prefix + str(idx)

        self.logger = None

        self.main_output_dir = self.making_dir(self.opt.save_dir, str(self.idx + 1), self.dir_name)

        self.b_keep_hold = Value(ctypes.c_bool, True)

    def run_begin(self) -> None:
        self.set_logger_file_handler(self.name + self.log_name, self.main_output_dir)
        self.logger.info(f'This is {self.name} Process')

    def process_run_action(self) -> None:
        self.b_keep_hold.value = False

    def hold_loop_action(self) -> None:
        ...

    def run_action(self) -> None:
        ...

    def run_end(self) -> None:
        ...

    def run(self) -> None:
        super(BaseProcess, self).run()

        self.logger = ALL_LoggerContainer.add_logger(self.name)
        ALL_LoggerContainer.add_stream_handler(self.name)
        log_level = 'debug' if self.opt.debug else 'info'
        ALL_LoggerContainer.set_logger_level(self.name, log_level)
        self.logger.info(f'set log level to {log_level}')

        self.run_begin()

        while self.b_keep_hold.value:
            self.hold_loop_action()

        self.run_action()

        self.run_end()

    @staticmethod
    def making_dir(*args) -> str:
        new_dir = os.path.join(*args)
        mkdir_if_missing(new_dir)

        return str(new_dir)

    @staticmethod
    def check_save_type(save_list: list) -> None:
        duplicates = []
        for element in save_list:
            if element not in wr.E_text_result_type:
                duplicates.append(element)
        if duplicates:
            alias_details = ', '.join(["%s" % name for name in duplicates])
            raise ValueError('Not valid save type: %s' % alias_details)

    def set_logger_file_handler(self, log_name: str, log_dir: str) -> None:
        ALL_LoggerContainer.add_file_handler(self.name, log_name, log_dir)

    def save_result_to_file(self, output_dir: str, result: dict):
        for each_type in self.save_type:
            # self.logger.info(f'Saving {each_type.name} result in {self.main_output_dir}')
            wr.write_results_to_text(
                output_dir,
                result,
                each_type
            )


class ProducerProcess(BaseProcess):
    def __init__(
            self,
            producer_bucket: ProducerBucket,
            *args,
            **kwargs,
    ) -> None:
        super(ProducerProcess, self).__init__(*args, **kwargs,)
        self.Bucket = producer_bucket


class ConsumerProcess(BaseProcess):
    def __init__(
            self,
            producer_bucket: ProducerBucket,
            output_type: str,
            data_shape: tuple,
            *args,
            **kwargs,
    ) -> None:
        super(ConsumerProcess, self).__init__(*args, **kwargs,)
        self.ProducerBucket = producer_bucket
        self.OutputPort = ConsumerOutputPort(self.opt, output_type, data_shape)


@unique
class EMultiprocess(Enum):
    ImageLoader = 1
    Tracker = 2
    Predictor = 3
    IndiPost = 4
    GlobalMatching = 5
    GlobalPost = 6


from .individual_process.MP_ImageLoader import ImageLoaderProcess
from .individual_process.MP_Tracker import TrackerProcess
from .individual_process.MP_PathPredict import PathPredictProcess
from .individual_process.MP_IndiPost import IndividualPostProcess


essential_process_indi_pre = [ImageLoaderProcess, TrackerProcess]

process_factory_indi_option = {
    EMultiprocess.Predictor.name: PathPredictProcess,
}

essential_process_indi_post = [IndividualPostProcess, ]

from .global_process.MP_GlobalIdMatch import GlobalIdMatchProcess
from .global_process.MP_GlobalPost import GlobalPostProcess

essential_process_global_pre = []

process_factory_global_option = {
    EMultiprocess.GlobalMatching.name: GlobalIdMatchProcess
}

essential_process_global_post = [GlobalPostProcess]

process_factory = {
    EMultiprocess.ImageLoader: ImageLoaderProcess,
    EMultiprocess.Tracker: TrackerProcess,
    EMultiprocess.Predictor: PathPredictProcess,
}
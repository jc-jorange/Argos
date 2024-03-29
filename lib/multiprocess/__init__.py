import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.tracker.utils.utils import mkdir_if_missing
import lib.postprocess.utils.write_result as wr
from lib.multiprocess.SharedMemory import SharedContainer


class BaseProcess(Process):
    prefix = ''
    dir_name = ''
    log_name = ''
    save_type = []

    def __init__(self,
                 shared_container: SharedContainer,
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
        self.final_save()

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

    def final_save(self):
        for each_type in self.save_type:
            self.logger.info(f'Saving {each_type.name} result in {self.main_output_dir}')
            wr.write_results_to_text(
                self.main_output_dir,
                self.all_frame_results,
                each_type
            )

@unique
class EMultiprocess(Enum):
    ImageLoader = 1
    Tracker = 2
    Predictor = 3
    IndiPost = 4
    GlobalMatching = 5


from .MP_ImageLoader import ImageLoaderProcess
from .MP_Tracker import TrackerProcess
from .MP_PathPredict import PathPredictProcess
from .MP_IndiPost import IndividualPostProcess


process_factory = {
    EMultiprocess.ImageLoader: ImageLoaderProcess,
    EMultiprocess.Tracker: TrackerProcess,
    EMultiprocess.Predictor: PathPredictProcess,
}

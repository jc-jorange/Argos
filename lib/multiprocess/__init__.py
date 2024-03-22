import os
from multiprocessing import Process
from enum import Enum, unique

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.multiprocess.Shared import ESharedDictType
from lib.tracker.utils.utils import mkdir_if_missing


@unique
class EMultiprocess(Enum):
    ImageReceiver = 1
    Tracker = 2
    Predictor = 3
    GlobalMatching = 4


class BaseProcess(Process):
    prefix = ''

    def __init__(self,
                 idx: int,
                 opt: opts,
                 container_shared_dict: {},
                 ) -> None:
        super(BaseProcess, self).__init__()

        self.idx = idx
        self.opt = opt
        self.container_shared_dict = container_shared_dict

        self.name = self.prefix + str(idx)

        self.logger = None

        self.main_output_dir = None

        self.b_keep_hold = True

    def run_begin(self) -> None:
        ...

    def process_run_action(self) -> None:
        self.b_keep_hold = False

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
        ALL_LoggerContainer.set_logger_level(self.name, 'debug' if self.opt.debug else 'info')
        self.logger.info('set log level from "debug" to "info" ')

        self.run_begin()

        while self.b_keep_hold:
            self.hold_loop_action()

        self.run_action()

        self.run_end()

    @staticmethod
    def making_dir(root: str, name: str) -> (str, bool):
        new_dir = os.path.join(root, name)
        mkdir_if_missing(new_dir)

        return str(new_dir), os.path.exists(new_dir)

    def making_process_main_save_dir(self, name: str) -> bool:
        self.main_output_dir, bExists = self.making_dir(self.opt.save_dir, name + str(self.idx + 1))
        return bExists

    def set_logger_file_handler(self, log_name: str, log_dir: str) -> None:
        ALL_LoggerContainer.add_file_handler(self.name, log_name, log_dir)

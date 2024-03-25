import ctypes
import os
from multiprocessing import Process, Queue, Value
from enum import Enum, unique
from typing import Dict

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.tracker.utils.utils import mkdir_if_missing
from Main import EQueueType

FRAME_DIR_NAME = 'frame'


@unique
class EMultiprocess(Enum):
    ImageReceiver = 1
    Tracker = 2
    Predictor = 3
    IndiPost = 4
    GlobalMatching = 5


class BaseProcess(Process):
    prefix = ''
    dir_name = ''

    def __init__(self,
                 idx: int,
                 opt: opts,
                 container_queue: Dict[EQueueType, Dict[int, Queue]],
                 container_result_dict: Dict[EMultiprocess, Dict[int, any]],
                 end_run_flag_shared_value: Value
                 ) -> None:
        super(BaseProcess, self).__init__()

        self.idx = idx
        self.opt = opt
        self.container_queue = container_queue
        self.container_result = container_result_dict

        self.name = self.prefix + str(idx)

        self.logger = None

        self.main_output_dir = self.making_dir(self.opt.save_dir, str(self.idx + 1), self.dir_name)

        self.end_run_flag = end_run_flag_shared_value
        self.b_keep_hold = Value(ctypes.c_bool, True)

    def run_begin(self) -> None:
        ...

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
        ALL_LoggerContainer.set_logger_level(self.name, 'debug' if self.opt.debug else 'info')
        self.logger.info('set log level from "debug" to "info" ')

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

    def set_logger_file_handler(self, log_name: str, log_dir: str) -> None:
        ALL_LoggerContainer.add_file_handler(self.name, log_name, log_dir)

import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique

from src.utils.logger import ALL_LoggerContainer
from src.multiprocess_pipeline.workers.tracker.utils.utils import mkdir_if_missing


name_Process_Result_dir = 'result'


@unique
class E_pipeline_branch(Enum):
    producer = 1
    consumer = 2
    post = 3
    static_shared_value = 4


class BaseProcess(Process):
    prefix: str = ''
    dir_name: str = ''
    log_name: str = ''

    b_save_in_index: bool = False

    shared_data: dict = {}

    def __init__(self,
                 data_hub,
                 pipeline_name: str,
                 opt,
                 ) -> None:
        super(BaseProcess, self).__init__()

        self.data_hub = data_hub
        self.pipeline_name = pipeline_name
        self.opt = opt

        self.main_save_dir = self.making_dir(self.opt.save_dir, self.pipeline_name, self.dir_name)
        self.results_save_dir = self.making_dir(self.main_save_dir, name_Process_Result_dir)
        if self.b_save_in_index:
            self.making_dir(self.results_save_dir, self.pipeline_name)
            self.results_save_dir = {self.pipeline_name: self.results_save_dir}

        self.name = self.prefix + self.pipeline_name

        self.logger = None

        self._b_hold_run = Value(ctypes.c_bool, True)
        self._b_finished = Value(ctypes.c_bool, False)

    def run_begin(self) -> None:
        ALL_LoggerContainer.add_file_handler(self.name, self.name + self.log_name, self.main_save_dir)
        self.logger.info(f'This is {self.name} Process')

    def process_run_action(self) -> None:
        self._b_hold_run.value = False

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

        while self._b_hold_run.value:
            self.hold_loop_action()

        self.run_action()

        self._b_finished.value = True

        self.run_end()

    def run_hold_state(self) -> bool:
        return self._b_hold_run.value

    def ready_finished(self) -> bool:
        return self._b_finished

    @staticmethod
    def making_dir(*args) -> str:
        new_dir = os.path.join(*args)
        mkdir_if_missing(new_dir)

        return str(new_dir)

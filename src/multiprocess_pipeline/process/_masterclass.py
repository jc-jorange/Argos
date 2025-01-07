import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique
import ntplib
import time

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
                 pipeline_index: int,
                 opt,
                 output_to_screen=True,
                 ) -> None:
        super(BaseProcess, self).__init__()

        self.data_hub = data_hub
        self.pipeline_name = pipeline_name
        self.pipeline_index = pipeline_index
        self.opt = opt
        self.output_to_screen = output_to_screen

        self.main_save_dir = self.making_dir(self.opt.save_dir, self.pipeline_name, self.dir_name)
        self.results_save_dir = self.making_dir(self.main_save_dir, name_Process_Result_dir)
        if self.b_save_in_index:
            self.making_dir(self.results_save_dir, self.pipeline_name)
            self.results_save_dir = {self.pipeline_name: self.results_save_dir}

        self.name = self.prefix + self.pipeline_name

        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request("pool.ntp.org")
            time_base_ntp = response.tx_time
        except:
            time_base_ntp = time.time()
        time_base_local = time.time()
        self.dt_base = time_base_ntp - time_base_local
        self.timestamp = 0

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
        if self.output_to_screen:
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

    def get_current_timestamp(self) -> int:
        self.timestamp = int((time.time() + self.dt_base) * 1000)
        return self.timestamp

    @staticmethod
    def making_dir(*args) -> str:
        new_dir = os.path.join(*args)
        mkdir_if_missing(new_dir)

        return str(new_dir)

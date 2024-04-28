import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.tracker.utils.utils import mkdir_if_missing
import lib.postprocess.utils.write_result as wr
from lib.multiprocess.SharedMemory import ProducerHub, ConsumerOutputPort
from lib.multiprocess.SharedMemory import E_SharedSaveType

Process_Result_dir = 'result'


@unique
class E_Result_Dir_Type(Enum):
    Index = 1


class BaseProcess(Process):
    prefix = ''
    dir_name = ''
    log_name = ''
    save_type = []
    save_dir_type: E_Result_Dir_Type = None

    def __init__(self,
                 producer_result_hub: ProducerHub,
                 idx: int,
                 opt: opts,
                 ) -> None:
        super(BaseProcess, self).__init__()
        self.check_save_type(self.save_type)

        self.producer_result_hub = producer_result_hub
        self.idx = idx
        self.opt = opt

        self.results_to_save = wr.S_default_save
        self.main_save_dir = self.making_dir(self.opt.save_dir, str(self.idx), self.dir_name)
        self.results_save_dir = self.making_dir(self.main_save_dir, Process_Result_dir)

        self.name = self.prefix + str(idx)

        self.logger = None

        self._b_hold_run = Value(ctypes.c_bool, True)

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

        self.run_end()

    def run_hold_state(self) -> bool:
        return self._b_hold_run.value

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

    def save_result_to_file(self, output_dir: str, result: dict) -> bool:
        b_save_result = False
        for each_type in self.save_type:
            # self.logger.info(f'Saving {each_type.name} result in {self.main_output_dir}')
            wr.write_results_to_text(
                output_dir,
                result,
                each_type
            )
            b_save_result = True
        return b_save_result


class ProducerProcess(BaseProcess):
    def __init__(
            self,
            *args,
            **kwargs,
    ) -> None:
        super(ProducerProcess, self).__init__(*args, **kwargs,)


class ConsumerProcess(BaseProcess):
    def __init__(
            self,
            output_type: E_SharedSaveType,
            data_shape: tuple,
            last_process_port: ConsumerOutputPort = None,
            *args,
            **kwargs,
    ) -> None:
        super(ConsumerProcess, self).__init__(*args, **kwargs,)
        self.output_port = ConsumerOutputPort(self.opt, output_type, data_shape)
        self.last_process_port = last_process_port


class PostProcess(BaseProcess):
    def __init__(
            self,
            # output_type: E_SharedSaveType,
            # data_shape: tuple,
            *args,
            **kwargs,
    ) -> None:
        super(PostProcess, self).__init__(*args, **kwargs,)
        # self.output_port = ConsumerOutputPort(self.opt, output_type, data_shape)

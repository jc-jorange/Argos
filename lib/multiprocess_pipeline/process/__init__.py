import ctypes
import os
from multiprocessing import Process, Value
from enum import Enum, unique

from lib.opts import opts
from lib.utils.logger import ALL_LoggerContainer
from lib.multiprocess_pipeline.workers.tracker.utils.utils import mkdir_if_missing
import lib.multiprocess_pipeline.workers.postprocess.utils.write_result as wr
from lib.multiprocess_pipeline.SharedMemory import format_SharedDataInfo

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
                 opt: opts,
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
    def check_shared_data(shared_data: dict) -> bool:
        l_wrong_num = []
        l_wrong_position = []
        l_wrong_element = []
        for k, v in shared_data.items():
            if not isinstance(v, tuple):
                v = tuple(v)
                shared_data[k] = v
            if len(v) != len(format_SharedDataInfo):
                l_wrong_num.append(k)
                continue
            for e in v:
                if e not in format_SharedDataInfo:
                    l_wrong_element.append(k)
                    break
                if v.index(e) != format_SharedDataInfo.index(e):
                    l_wrong_position.append(k)
                    break

        if l_wrong_num or l_wrong_position or l_wrong_element:
            raise ValueError(f'process shared data check not pass in '
                             f'wrong element number: {l_wrong_num}, '
                             f'wrong position: {l_wrong_position}, '
                             f'wrong element: {l_wrong_element}')
        else:
            return True


class ProducerProcess(BaseProcess):
    def __init__(self,
                 *args, **kwargs) -> None:
        super(ProducerProcess, self).__init__(*args, **kwargs,)


from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType, E_OutputPortDataType, Struc_ConsumerOutputPort


class ConsumerProcess(BaseProcess):
    output_type: E_SharedSaveType = E_SharedSaveType.Queue
    output_data_type: E_OutputPortDataType = E_OutputPortDataType.Default
    output_buffer: int = 8
    data_shape: tuple = (1,)
    results_save_type = [wr.E_text_result_type.raw]

    def __init__(self,
                 last_process_port: Struc_ConsumerOutputPort = None,
                 *args, **kwargs) -> None:
        super(ConsumerProcess, self).__init__(*args, **kwargs,)
        self.check_results_save_type(self.results_save_type)

        self.output_port = Struc_ConsumerOutputPort(self.opt,
                                                    self.output_type,
                                                    self.output_data_type,
                                                    self.data_shape)
        self.last_process_port = last_process_port

    @staticmethod
    def check_results_save_type(save_type: list) -> bool:
        duplicates = []
        for element in save_type:
            if element not in wr.E_text_result_type:
                duplicates.append(element)
        if duplicates:
            alias_details = ', '.join(["%s" % name for name in duplicates])
            raise ValueError('Not valid save type: %s' % alias_details)
        return True

    def save_result_to_file(self, output_dir: str, result: dict) -> bool:
        b_save_result = False
        for each_type in self.results_save_type:
            wr.write_results_to_text(
                output_dir,
                result,
                each_type
            )
            b_save_result = True
        return b_save_result

    def run_end(self) -> None:
        super(ConsumerProcess, self).run_end()
        self.output_port.clear()


class PostProcess(BaseProcess):
    def __init__(self,
                 *args, **kwargs) -> None:
        super(PostProcess, self).__init__(*args, **kwargs,)


from lib.multiprocess_pipeline.process.producer import factory_process_producer
from lib.multiprocess_pipeline.process.consumer import factory_process_consumer
from lib.multiprocess_pipeline.process.post import factory_process_post

factory_process_all = {
    E_pipeline_branch.producer.name: factory_process_producer,
    E_pipeline_branch.consumer.name: factory_process_consumer,
    E_pipeline_branch.post.name: factory_process_post,
}

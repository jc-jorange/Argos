import numpy
from multiprocessing import queues

from . import ConsumerProcess
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType, \
    E_PipelineSharedDataName
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from src.multiprocess_pipeline.workers.data_sender import factory_data_sender


class DataSendProcess(ConsumerProcess):
    prefix = 'Argos-SubProcess-Global_Data_Send_'
    dir_name = 'Data_Send'
    log_name = 'Data_Send_Log'
    save_type = [wr.E_text_result_type.raw]
    b_save_in_index = True

    output_type = E_SharedSaveType.Queue
    output_data_type = E_OutputPortDataType.CameraTrack
    output_shape = (1,)

    def __init__(self,
                 sender_name,
                 target,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sender_name = sender_name
        self.target = target

        if self.last_process_port.data_type != E_OutputPortDataType.CameraTrack:
            raise TypeError('Connect last consumer process output data type not fit')

        self.sender = None

    def run_begin(self) -> None:
        super(DataSendProcess, self).run_begin()
        self.logger.info(f'Creating data sender {self.sender_name}')
        self.sender = factory_data_sender[self.sender_name](
            target=self.target,
        )

    def run_action(self) -> None:
        super(DataSendProcess, self).run_action()
        self.logger.info('Start sending data')

        # Get this pipeline producer is alive
        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        while hub_b_loading.value:
            frame = 0
            subframe = 0
            try:
                # Get data from last consumer
                last_result = self.last_process_port.read()
                frame = last_result[0]
                subframe = last_result[1]
                data = last_result[-1]
                self.logger.debug(f'Get last port data @ frame:{frame}, subframe:{subframe}')
                # Confirm if we get a new frame result
            except queues.Empty:
                continue

            if self.sender.send_data(data):
                if frame % 10 == 0 and subframe % 100 == 0:
                    self.logger.info(f'Send data SUCCESS to {self.target} @ frame:{frame}, subframe:{subframe}')
            else:
                self.logger.info(f'Send data FAIL to {self.target} @ frame:{frame}, subframe:{subframe}')

            if self.output_port.size() < self.output_buffer:
                self.output_port.send((frame, subframe, data))
                self.logger.debug(f'Pass data to next')
            else:
                self.logger.debug(f'Output port over {self.output_buffer} buffer size')

    def run_end(self) -> None:
        super(DataSendProcess, self).run_end()
        self.logger.info('-' * 5 + 'Data Sending Finished' + '-' * 5)

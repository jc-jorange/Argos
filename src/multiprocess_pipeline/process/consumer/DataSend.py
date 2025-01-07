import time

import numpy

from . import ConsumerProcess
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType, \
    E_PipelineSharedDataName
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from src.multiprocess_pipeline.workers.data_sender import factory_data_sender
from src.multiprocess_pipeline.workers.data_filter import factory_data_filter


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
                 filter_name,
                 send_target,
                 *args,
                 filter_target=(),
                 with_flag=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sender_name = sender_name
        self.filter_name = filter_name
        self.send_target = send_target
        self.filter_target = filter_target
        self.with_flag = with_flag

        if self.last_process_port.data_type != E_OutputPortDataType.CameraTrack:
            raise TypeError('Connect last consumer process output data type not fit')

        self.sender = None
        self.filter = None

    def run_begin(self) -> None:
        super(DataSendProcess, self).run_begin()
        self.logger.info(f'Creating data sender {self.sender_name}')
        self.sender = factory_data_sender[self.sender_name](
            target=self.send_target,
            with_flag=self.with_flag,
        )
        self.filter = factory_data_filter[self.filter_name](
            target=self.filter_target,
        )

    def run_action(self) -> None:
        super(DataSendProcess, self).run_action()
        self.logger.info('Start sending data')

        # Get this pipeline producer is alive
        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]

        while hub_b_loading.value:
            # Get data from last consumer
            last_result = self.last_process_port.read()

            if last_result is None:
                continue

            timestamp_image = last_result[0]
            frame = last_result[1]
            subframe = last_result[2]
            data = last_result[-1]

            data_filtered = self.filter.filter_data(data)
            if not isinstance(data_filtered, numpy.ndarray):
                continue

            self.logger.debug(f'Get last port data @ frame:{frame}, subframe:{subframe}')
            # Confirm if we get a new frame result

            if self.sender.send_action(self.get_current_timestamp(), data_filtered):
                local_timestamp = (time.time() + self.dt_base) * 1000
                print(f'dt camera image send: {local_timestamp - timestamp_image}')
                if frame % 10 == 0 and subframe % 10 == 0:
                    self.logger.info(f'Send data SUCCESS to {self.send_target} @ frame:{frame}, subframe:{subframe}')
            else:
                self.logger.info(f'Send data FAIL to {self.send_target} @ frame:{frame}, subframe:{subframe}')

            if self.output_port.size() >= self.output_buffer:
                self.logger.debug(f'Output port over {self.output_buffer} buffer size')
                self.output_port.read()
            self.output_port.send((frame, subframe, data))
            self.logger.debug(f'Pass data to next')
            time.sleep(0.001)

    def run_end(self) -> None:
        super(DataSendProcess, self).run_end()
        self.logger.info('-' * 5 + 'Data Sending Finished' + '-' * 5)

import numpy

from . import ConsumerProcess
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType, \
    E_PipelineSharedDataName
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr
from src.multiprocess_pipeline.workers.data_smoother import factory_data_smoother


class DataSmoothProcess(ConsumerProcess):
    prefix = 'Argos-SubProcess-Data_Smooth_'
    dir_name = 'Data_Smooth'
    log_name = 'Data_Smooth_Log'
    save_type = [wr.E_text_result_type.raw]
    b_save_in_index = False

    output_type = E_SharedSaveType.Queue
    output_data_type = E_OutputPortDataType.CameraTrack
    output_shape = (1,)

    def __init__(self,
                 smoother_name: str,
                 smoother_kwargs={},
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.smoother_name = smoother_name
        self.smoother_kwargs = smoother_kwargs

        if self.last_process_port.data_type != E_OutputPortDataType.CameraTrack:
            raise TypeError('Connect last consumer process output data type not fit')

        self.smoother = None

    def run_begin(self) -> None:
        super(DataSmoothProcess, self).run_begin()
        self.logger.info(f'Creating data smoother {self.smoother_name}')
        self.smoother = factory_data_smoother[self.smoother_name](**self.smoother_kwargs)

    def run_action(self) -> None:
        super(DataSmoothProcess, self).run_action()
        self.logger.info('Start sending data')

        # Get this pipeline producer is alive
        hub_b_loading = self.data_hub.dict_bLoadingFlag[self.pipeline_name]
        self.last_data = []

        while hub_b_loading.value:
            timestamp = 0
            frame = 0
            subframe = 0
            # Get data from last consumer
            last_result = self.last_process_port.read()

            if last_result is None:
                continue

            timestamp = last_result[0]
            frame = last_result[1]
            subframe = last_result[2]
            data = last_result[-1]

            if data is None:
                continue
            self.logger.debug(f'Get last port data @ frame:{frame}, subframe:{subframe}')
            # Confirm if we get a new frame result

            self.smoother.put_data(data)
            smoothed_data = self.smoother.smooth_action()
            if smoothed_data is not None:
                data = smoothed_data

                if frame % 10 == 0 and subframe % 10 == 0:
                    self.logger.info(f'Smooth data from {self.last_process_port.consumer_name} '
                                     f'@ frame:{frame}, subframe:{subframe}')

            if self.output_port.size() >= self.output_buffer:
                self.logger.debug(f'Output port over {self.output_buffer} buffer size')
                self.output_port.read()
            self.output_port.send((timestamp, frame, subframe, data))
            self.logger.debug(f'Pass data to next')

    def run_end(self) -> None:
        super(DataSmoothProcess, self).run_end()
        self.logger.info('-' * 5 + 'Data Smooth Finished' + '-' * 5)

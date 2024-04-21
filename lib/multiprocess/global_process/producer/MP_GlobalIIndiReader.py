import multiprocessing as mp

from lib.multiprocess import ProducerProcess
from lib.multiprocess.SharedMemory import E_ProducerOutputName_Indi
from lib.multiprocess.SharedMemory import E_ProducerOutputName_Global, E_ProducerOutputName_Global_PassThrough
from lib.multiprocess.SharedMemory import ProducerHub_Indi, ProducerHub_Global, ConsumerOutputPort


class IndiResultsReader(ProducerProcess):
    prefix = 'Argus-SubProcess-Global_Reader_'
    dir_name = 'global_reader'
    log_name = 'Global_Reader_Log'
    save_type = []

    def __init__(self,
                 indi_results: dict,
                 indi_results_hub: dict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.all_indi_results = indi_results
        self.all_indi_results_hub = indi_results_hub

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start read each camera result to global process')

        while self.producer_result_hub.output[E_ProducerOutputName_Global.bInputLoading].b_input_loading.value:
            v_i: ProducerHub_Indi
            self.producer_result_hub: ProducerHub_Global
            for k_i, v_i in self.all_indi_results_hub:
                camera_transform = v_i.output[E_ProducerOutputName_Indi.CameraTransform:]
                self.producer_result_hub.output_passthrough[k_i][
                    E_ProducerOutputName_Global_PassThrough.CameraTransformAll
                ].put(camera_transform)

            v_i_q: ConsumerOutputPort
            for k_i, v_i_q in self.all_indi_results:
                indi_result = v_i_q.output.get(block=False)
                self.producer_result_hub.output_passthrough[k_i][
                    E_ProducerOutputName_Global_PassThrough.PredictAll
                ].put(indi_result)

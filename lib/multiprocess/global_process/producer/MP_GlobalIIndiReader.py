from lib.multiprocess import ConsumerProcess
from lib.multiprocess.SharedMemory import E_ProducerOutputName_Global


class IndiResultsReader(ConsumerProcess):
    prefix = 'Argus-SubProcess-Global_Reader_'
    dir_name = 'global_reader'
    log_name = 'Global_Reader_Log'
    save_type = []

    def __init__(self,
                 indi_results: dict,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.all_indi_results = indi_results

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start read each camera result to global process')

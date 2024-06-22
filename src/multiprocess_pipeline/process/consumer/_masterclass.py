from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_OutputPortDataType, Struc_ConsumerOutputPort
import src.multiprocess_pipeline.workers.postprocess.utils.write_result as wr

from .. import BaseProcess


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
        self.logger.info(f'{self.name} start clearing')
        self.output_port.clear()
        self.logger.info(f'{self.name} clear over')

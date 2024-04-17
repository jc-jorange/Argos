import multiprocessing
import time
from typing import Type
from collections import defaultdict
import numpy

from lib.multiprocess import ConsumerProcess
from lib.matchor import BaseMatchor
from lib.postprocess.utils.write_result import convert_numpy_to_dict
from lib.postprocess.utils import write_result as wr


class IndiResultsReader(ConsumerProcess):
    prefix = 'Argus-SubProcess-Global_Reader_'
    dir_name = 'global_reader'
    log_name = 'Global_Reader_Log'
    save_type = []

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def run_action(self) -> None:
        super().run_action()
        self.logger.info('Start read each camera result to global process')

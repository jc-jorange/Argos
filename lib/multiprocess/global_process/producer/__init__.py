from enum import Enum, unique

from lib.multiprocess.global_process.producer.MP_GlobalIIndiReader import IndiResultsReader


@unique
class E_Global_Process_Producer(Enum):
    IndiResultsReader = 1


factory_global_process_producer = {
    E_Global_Process_Producer.IndiResultsReader.name: IndiResultsReader,
}

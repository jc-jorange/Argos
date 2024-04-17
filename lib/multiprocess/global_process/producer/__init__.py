from enum import Enum, unique

from lib.multiprocess.global_process.producer.MP_GlobalIIndiReader import IndiResultsReader


@unique
class E_Global_Process_Post(Enum):
    IndiResultsReader = 1


factory_global_process = {
    E_Global_Process_Post.IndiResultsReader.name: IndiResultsReader,
}

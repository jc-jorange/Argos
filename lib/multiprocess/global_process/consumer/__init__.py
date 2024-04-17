from enum import Enum, unique

from lib.multiprocess.global_process.consumer.MP_MultiCameraIdMatch import MultiCameraIdMatchProcess


@unique
class E_Global_Process_Consumer(Enum):
    MultiCameraIdMatchProcess = 1


factory_global_process_Consumer = {
    E_Global_Process_Consumer.GlobalMatching.name: MultiCameraIdMatchProcess,
}

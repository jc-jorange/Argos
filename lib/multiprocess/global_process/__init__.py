from enum import Enum, unique

from lib.multiprocess.global_process.consumer.MP_MultiCameraIdMatch import MultiCameraIdMatchProcess
from lib.multiprocess.global_process.post.MP_GlobalPost import GlobalPostProcess


@unique
class E_Global_Process(Enum):
    GlobalMatching = 1
    GlobalPost = 2


factory_global_process = {
    E_Global_Process.GlobalMatching.name: MultiCameraIdMatchProcess,
    E_Global_Process.GlobalPost.name: GlobalPostProcess,
}

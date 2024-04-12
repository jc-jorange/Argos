from enum import Enum, unique

from .MP_GlobalIdMatch import GlobalIdMatchProcess
from .MP_GlobalPost import GlobalPostProcess

@unique
class E_Global_Process(Enum):
    GlobalMatching = 1
    GlobalPost = 2

factory_global_process = {
    E_Global_Process.GlobalMatching.name: GlobalIdMatchProcess,
    E_Global_Process.GlobalPost.name: GlobalPostProcess,
}
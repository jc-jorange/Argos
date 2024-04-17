from enum import Enum, unique

from lib.multiprocess.global_process.post.MP_GlobalPost import GlobalPostProcess


@unique
class E_Global_Process_Post(Enum):
    GlobalPost = 1


factory_global_process = {
    E_Global_Process_Post.GlobalPost.name: GlobalPostProcess,
}

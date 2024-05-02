from enum import Enum, unique

from lib.multiprocess_pipeline.process_group.global_process.post.MP_GlobalPost import GlobalPostProcess


@unique
class E_Global_Process_Post(Enum):
    GlobalPost = 1


factory_global_process_post = {
    E_Global_Process_Post.GlobalPost.name: GlobalPostProcess,
}

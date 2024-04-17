from enum import Enum, unique

from lib.multiprocess.individual_process.post.MP_IndiPost import IndividualPostProcess


@unique
class E_Indi_Process_Post(Enum):
    IndiPost = 1


factory_indi_process_post = {
    E_Indi_Process_Post.IndiPost.name: IndividualPostProcess,
}
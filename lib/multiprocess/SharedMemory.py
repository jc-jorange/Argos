import multiprocessing as mp
from multiprocessing import shared_memory
from enum import Enum, unique
import numpy as np
import torch

from lib.model.model_config import E_model_general_info, model_general_info_default_dict


@unique
class EQueueType(Enum):
    LoadResultSend = 1
    TrackerResultSend = 2
    PredictResultSend = 3

@unique
class EResultType(Enum):
    TrackResult = 1
    PredictResult = 2
    MatchResult = 3


def store_in_shm(name: str, data) -> shared_memory.SharedMemory:
    shm = shared_memory.SharedMemory(name=name, create=True, size=data.nbytes)
    shm_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shm_data[:] = data[:]
    # there must always be at least one `SharedMemory` object open for it to not
    #  be destroyed on Windows, so we won't `shm.close()` inside the function,
    #  but rather after we're done with everything.
    return shm


def read_from_shm(name: str, shape, dtype) -> (shared_memory.SharedMemory, np.ndarray):
    shm = shared_memory.SharedMemory(name=name, create=False)
    shm_data = np.ndarray(shape, dtype, buffer=shm.buf)
    return shm, shm_data
    # we need to keep a reference of shm both so we don't
    #  segfault on shmData and so we can `close()` it.


class SharedContainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = ''
        self.resized_tensor = None

        self.reset(self.opt.net_input_shape, self.opt.device)

        self.b_input_loading = mp.Value('B', True)

        self.input_frame_id = mp.Value('I', 0)

        self.origin_shape = (mp.Value('I', 0), mp.Value('I', 0), mp.Value('I', 0))

        self.max_class = self.opt.model_config[E_model_general_info.max_classes_num.name] \
            if self.opt.model_config[E_model_general_info.max_classes_num.name] > model_general_info_default_dict[E_model_general_info.max_classes_num] \
            else model_general_info_default_dict[E_model_general_info.max_classes_num]
        self.max_object = self.opt.model_config[E_model_general_info.max_objects_num.name] \
            if self.opt.model_config[E_model_general_info.max_objects_num.name] > model_general_info_default_dict[E_model_general_info.max_objects_num] \
            else model_general_info_default_dict[E_model_general_info.max_objects_num]

        self.queue_dict = {QueueType: mp.Queue() for name, QueueType in EQueueType.__members__.items()}
        self.result_dict = {ResultType: mp.Array('f', self.max_class * self.max_object * 4, lock=False) for name, ResultType in EResultType.__members__.items()}

    def set_origin_shape(self, shape: tuple) -> None:
        self.origin_shape[0].value = shape[0]
        self.origin_shape[1].value = shape[1]
        self.origin_shape[2].value = shape[2]

    def get_origin_shape(self) -> tuple:
        c = self.origin_shape[0].value
        h = self.origin_shape[1].value
        w = self.origin_shape[2].value
        return tuple((c, h, w))

    def reset(self, shape: tuple, device: str) -> None:
        self.resized_tensor = torch.ones(shape, dtype=torch.float, device=device).unsqueeze(0)
        self.resized_tensor.share_memory_()

    def set_data(self, data: np.ndarray) -> None:
        self.resized_tensor[:] = torch.from_numpy(data).to(self.resized_tensor)

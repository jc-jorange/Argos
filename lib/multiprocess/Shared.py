from multiprocessing import shared_memory
import multiprocessing as mp
import numpy as np
from enum import Enum, unique


@unique
class ESharedDictType(Enum):
    Image_Input_List = 1
    Image_Current = 2
    Track = 3
    Predict = 4


NAME_shm_img = 'RecvImg_'


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


class SharedList:
    def __init__(self) -> None:
        self.mp_list = mp.Manager().list()

    def append(self, data) -> None:
        self.mp_list.append(data)

    def set(self, idx: int, data) -> None:
        self.mp_list[idx] = data

    def insert(self, idx: int, data) -> None:
        self.mp_list.insert(idx, data)

    def read(self, idx: int) -> any:
        return self.mp_list[idx]

    def pop(self, key: any) -> any:
        try:
            return self.mp_list.pop(key)
        except KeyError:
            return None


class SharedDict:
    def __init__(self) -> None:
        self.mp_dict = mp.Manager().dict()

    def set(self, key: any, data: any) -> None:
        self.mp_dict[key] = data

    def read(self, key: any) -> any:
        try:
            return self.mp_dict[key]
        except KeyError:
            return None

    def pop(self, key: any) -> any:
        try:
            return self.mp_dict.pop(key)
        except KeyError:
            return None

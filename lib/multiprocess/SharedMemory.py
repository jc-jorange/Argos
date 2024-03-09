import numpy
import numpy as np
from multiprocessing import shared_memory

NAME_shm_img = 'RecvImg_'

def store_in_shm(self, name: str, data):
    shm = shared_memory.SharedMemory(name=name, create=True, size=data.nbytes)
    shmData = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shmData[:] = data[:]
    #there must always be at least one `SharedMemory` object open for it to not
    #  be destroyed on Windows, so we won't `shm.close()` inside the function,
    #  but rather after we're done with everything.
    return shm

def read_from_shm(self, name: str, shape, dtype):
    shm = shared_memory.SharedMemory(name=name, create=False)
    shmData = np.ndarray(shape, dtype, buffer=shm.buf)
    return shm, shmData #we need to keep a reference of shm both so we don't
                        #  segfault on shmData and so we can `close()` it.


class InfoStruct:
    def __init__(self, shape: tuple, data_type: numpy.number):
        self.shape = shape
        self.data_type = data_type


class SharedMemory:
    def __init__(self, name: str, data):
        self.name = name
        self.shm = shm = shared_memory.SharedMemory(name=name, create=True, size=data.nbytes)
        shm_data = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
        shm_data[:] = data[:]

        self.info = InfoStruct(data.shape, data.dtype)

    def read_data(self):
        return np.ndarray(self.info.shape, self.info.data_type, buffer=self.shm.buf)


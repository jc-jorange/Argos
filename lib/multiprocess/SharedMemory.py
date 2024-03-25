from multiprocessing import shared_memory
import numpy as np


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

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


@unique
class E_SharedSaveType(Enum):
    Queue = 1
    Tensor = 2
    SharedArray_Int = 3
    SharedArray_Float = 4
    SharedValue_Int = 5
    SharedValue_Float = 6


@unique
class E_ProducerOutputName(Enum):
    FrameID = 1
    ImageOriginShape = 2
    ImageData = 3
    bInputLoading = 4


format_ProducerOutput = (E_SharedSaveType, tuple, int)
dict_ProducerOutput = {
    E_ProducerOutputName.FrameID: (E_SharedSaveType.SharedValue_Int, (1,), 0),
    E_ProducerOutputName.ImageOriginShape: (E_SharedSaveType.SharedArray_Int, (3,), 0),
    E_ProducerOutputName.ImageData: (E_SharedSaveType.Queue, (1,), 0),
    E_ProducerOutputName.bInputLoading: (E_SharedSaveType.SharedValue_Int, (1,), 1)
}


class ProducerBucket:
    def __init__(self, opt):
        self.opt = opt
        self.device = ''

        if self.opt.realtime:
            dict_ProducerOutput[E_ProducerOutputName.ImageData] = (E_SharedSaveType.Tensor, self.opt.net_input_shape, 0)
        self.output = {}
        for k, v in dict_ProducerOutput.items():
            self.output[k] = self.generate_output_value(v)

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

    def generate_output_value(self, output_format: format_ProducerOutput) -> any:
        data_type: E_SharedSaveType = output_format[0]
        data_shape: tuple = output_format[1]
        default_value: int = output_format[2]

        if data_type == E_SharedSaveType.Queue:
            output_value = mp.Queue()
        elif data_type == E_SharedSaveType.SharedArray_Int:
            output_value = mp.Array('i', sum(data_shape))
            output_value[:] = default_value
        elif data_type == E_SharedSaveType.Tensor:
            if default_value:
                output_value = torch.ones(data_shape, dtype=torch.float, device=self.opt.device)
            else:
                output_value = torch.empty(data_shape, dtype=torch.float, device=self.opt.device)
            output_value.share_memory_()
        elif data_type == E_SharedSaveType.SharedValue_Int:
            output_value = mp.Value('i', default_value)
        else:
            raise ValueError(f'Wrong producer output type as {data_type}')

        return output_value


class ConsumerOutputPort:
    def __init__(self, opt, output_type: str, data_shape: tuple):
        self.opt = opt

        self._output_type = E_SharedSaveType[output_type]
        self._data_shape = None
        self.output = None

        self.reset(data_shape)

    def reset(self, data_shape: tuple):
        del self.output

        if self._output_type == E_SharedSaveType.Queue:
            self.output = mp.Queue()
        elif self._output_type == E_SharedSaveType.SharedArray_Int:
            self.output = mp.Array('f', sum(data_shape), lock=False)
        elif self._output_type == E_SharedSaveType.Tensor:
            self.output = torch.ones(data_shape, dtype=torch.float, device=self.opt.device)
            self.output.share_memory_()

        self._data_shape = data_shape

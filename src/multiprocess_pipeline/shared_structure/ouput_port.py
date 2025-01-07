import multiprocessing as mp
import numpy as np
import torch
from enum import Enum, unique

from .shared_data import E_SharedSaveType


@unique
class E_OutputPortDataType(Enum):
    Default = 1
    CameraTrack = 2


dict_OutputPortDataType = {
    E_OutputPortDataType.Default.name: any,
    E_OutputPortDataType.CameraTrack.name: (int, int, np.ndarray),
}


class Struc_ConsumerOutputPort:
    def __init__(self,
                 opt,
                 consumer_name: str,
                 output_type: E_SharedSaveType,
                 data_type: E_OutputPortDataType,
                 data_shape: tuple
                 ):
        self.opt = opt

        self.consumer_name = consumer_name
        self.output_type = output_type
        self.data_type = data_type
        self.data_shape = None
        self.output = None

        self.reset(data_shape)

    def reset(self, data_shape: tuple) -> None:
        del self.output

        if self.output_type == E_SharedSaveType.Queue:
            self.output = mp.Queue()
        elif self.output_type == E_SharedSaveType.SharedArray_Float:
            self.output = mp.Array('f', sum(data_shape), lock=False)
        elif self.output_type == E_SharedSaveType.Tensor:
            self.output = torch.ones(data_shape, dtype=torch.float, device=self.opt.device)
            self.output.share_memory_()

        self.data_shape = data_shape

    def read(self) -> np.ndarray:
        result = None

        if self.output_type == E_SharedSaveType.Queue:
            self.output: mp.Queue
            if self.output.empty():
                result = None
            else:
                try:
                    result = self.output.get(block=False)
                except mp.queues.Empty:
                    result = None

        elif self.output_type == E_SharedSaveType.SharedArray_Float:
            self.output: mp.Array
            result = np.frombuffer(self.output.get_obj())
            result.reshape(self.data_shape)

        elif self.output_type == E_SharedSaveType.Tensor:
            self.output: torch.Tensor
            result = self.output.cpu().numpy()

        return result

    def send(self, data) -> bool:
        b_result = False

        if self.size() > 0:
            self.read()

        if self.output_type == E_SharedSaveType.Queue:
            self.output: mp.Queue
            self.output.put(data, block=False)
            b_result = True
        elif self.output_type == E_SharedSaveType.SharedArray_Float:
            self.output: mp.Array
            to_send = np.asarray(data)
            if to_send.shape == self.data_shape:
                to_send = to_send.flatten()
                self.output[:] = to_send[:]
                b_result = True
        elif self.output_type == E_SharedSaveType.SharedValue_Float:
            self.output: mp.Value
            self.output.value = data
            b_result = True
        elif self.output_type == E_SharedSaveType.Tensor:
            if isinstance(data, torch.Tensor):
                self.output[:] = data[:]
            else:
                to_send = np.asarray(data)
                self.output[:] = to_send[:]
            b_result = True

        return b_result

    def size(self) -> int:
        if self.output_type == E_SharedSaveType.Queue:
            self.output: mp.Queue
            return self.output.qsize()
        else:
            return 1

    def clear(self) -> None:
        if self.output_type == E_SharedSaveType.Queue:
            self.output: mp.Queue
            while self.output.qsize() > 0:
                try:
                    self.output.get(block=False)
                except RuntimeError:  # CUDA error: invalid device context
                    pass
            self.output.close()
        else:
            del self.output

        torch.cuda.empty_cache()

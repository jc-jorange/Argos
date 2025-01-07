import multiprocessing as mp
import numpy as np
import torch
from enum import Enum, unique
import math
from collections import Iterable
import queue


@unique
class E_SharedSaveType(Enum):
    Queue = 1
    Tensor = 2
    SharedArray_Int = 3
    SharedArray_Float = 4
    SharedValue_Int = 5
    SharedValue_Float = 6


@unique
class E_SharedDataFormat(Enum):
    data_type = 1
    data_shape = 2
    data_value = 3


dict_SharedDataInfoFormat = {
    E_SharedDataFormat.data_type.name: E_SharedSaveType,
    E_SharedDataFormat.data_shape.name: tuple,
    E_SharedDataFormat.data_value.name: any,
}


class Struc_SharedData:
    _data = None
    data_type = None
    data_shape = ()
    device = 'cpu'

    def _generate_output_value(self, output_format: dict_SharedDataInfoFormat) -> any:
        data_type = output_format[E_SharedDataFormat.data_type.name]
        data_type = data_type if isinstance(data_type, E_SharedSaveType) else E_SharedSaveType[data_type]
        self.data_type = data_type
        data_shape = output_format[E_SharedDataFormat.data_shape.name]
        self.data_shape = tuple(data_shape)
        default_value = output_format[E_SharedDataFormat.data_value.name]

        if data_type == E_SharedSaveType.Queue:
            output_value = mp.Queue()
            if default_value:
                output_value.put(default_value)

        elif data_type == E_SharedSaveType.Tensor:
            if isinstance(default_value, Iterable):
                default_value = np.asarray(default_value)
                output_value = torch.from_numpy(default_value)
                output_value = output_value.type(torch.float)
                output_value = output_value.to(self.device)
            else:
                if default_value:
                    output_value = torch.ones(data_shape, dtype=torch.float, device=self.device)
                    output_value = output_value * default_value
                else:
                    output_value = torch.empty(data_shape, dtype=torch.float, device=self.device)
                output_value.share_memory_()

        elif data_type == E_SharedSaveType.SharedArray_Int:
            total_size = math.prod(data_shape)
            if isinstance(default_value, Iterable):
                default_value_list = np.asarray(default_value)
                default_value_list = default_value_list.flatten()
            else:
                default_value_list = [default_value] * total_size
            output_value = mp.Array('i', total_size)
            output_value[:] = default_value_list[:]
        elif data_type == E_SharedSaveType.SharedArray_Float:
            total_size = math.prod(data_shape)
            if isinstance(default_value, Iterable):
                default_value_list = np.asarray(default_value)
                default_value_list = default_value_list.flatten()
            else:
                default_value_list = [default_value] * total_size
            output_value = mp.Array('f', total_size)
            output_value[:] = default_value_list[:]

        elif data_type == E_SharedSaveType.SharedValue_Int:
            assert not isinstance(default_value, Iterable), f'iterable default value apply to single value'
            output_value = mp.Value('i', default_value)
        elif data_type == E_SharedSaveType.SharedValue_Float:
            assert not isinstance(default_value, Iterable), f'iterable default value apply to single value'
            output_value = mp.Value('f', default_value)

        else:
            raise ValueError(f'Wrong producer output type as {data_type}')

        return output_value

    def __init__(self,
                 device: str,
                 output_format: dict_SharedDataInfoFormat):
        self._bBeenSet = mp.Value('i', 1)
        self.reset(output_format, device)

    def reset(self, data_format, device):
        self.device = device
        self._data = self._generate_output_value(data_format)

    def set(self, data_set) -> None:
        if self.data_type == E_SharedSaveType.Queue:
            self._data.put(data_set)

        elif self.data_type == E_SharedSaveType.Tensor:
            new_data = data_set
            if isinstance(new_data, torch.Tensor):
                self._data[:] = new_data.clone()
                self._data.to(new_data.device)
                self.device = new_data.device
            elif isinstance(new_data, np.ndarray):
                self._data[:] = torch.from_numpy(new_data).clone()
                self._data.to(self.device)
                self._data.share_memory_()
            else:
                if isinstance(new_data, Iterable):
                    new_data = np.asarray(data_set)
                else:
                    new_data = np.ones()
                    new_data = new_data * data_set
                self._data[:] = torch.from_numpy(new_data).clone()
                self._data.to(self.device)
                self._data.share_memory_()
            self.data_shape = tuple(self._data.data.shape)

        elif self.data_type == E_SharedSaveType.SharedArray_Int or \
                self.data_type == E_SharedSaveType.SharedArray_Float:
            data_set = np.asarray(data_set)
            data_set = data_set.flatten()
            self._data[:] = data_set[:]

        elif self.data_type == E_SharedSaveType.SharedValue_Int or \
                self.data_type == E_SharedSaveType.SharedValue_Float:
            self._data.value = data_set

        self._bBeenSet.value = 1

    def get(self) -> any:
        if self.data_type == E_SharedSaveType.Queue:
            if self._data.empty():
                self._bBeenSet.value = 0
                return None
            else:
                try:
                    re_data = self._data.get(block=False)
                    return re_data
                except RuntimeError:  # CUDA error: invalid device context
                    return None
                except queue.Empty:
                    return None

        elif self.data_type == E_SharedSaveType.Tensor:
            if self._bBeenSet.value:
                self._bBeenSet.value = 0
                return self._data.data.clone()
            else:
                return None

        elif self.data_type == E_SharedSaveType.SharedArray_Int or \
                self.data_type == E_SharedSaveType.SharedArray_Float:
            if self._bBeenSet.value:
                data = self._data[:]
                data = np.asarray(data)
                data = data.reshape(self.data_shape)
                self._bBeenSet.value = 0
                return data
            else:
                return None

        elif self.data_type == E_SharedSaveType.SharedValue_Int or \
                self.data_type == E_SharedSaveType.SharedValue_Float:
            if self._bBeenSet.value:
                re_data = self._data.value
                self._bBeenSet.value = 0
                return re_data
            else:
                return None

        else:
            raise ValueError

    def clear(self) -> None:
        if self.data_type == E_SharedSaveType.Queue:
            self._data: mp.Queue
            while self._data.qsize() > 0:
                try:
                    self._data.get(block=False)
                except RuntimeError:  # CUDA error: invalid device context
                    pass
        else:
            del self._data

        torch.cuda.empty_cache()

    def size(self) -> int:
        if self.data_type == E_SharedSaveType.Queue:
            return self._data.qsize()
        else:
            return 1

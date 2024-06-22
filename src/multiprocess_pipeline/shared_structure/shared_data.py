import multiprocessing as mp
import numpy as np
import torch
from enum import Enum, unique
import math
from collections import Iterable


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
        self.device = device
        self._data = self._generate_output_value(output_format)

    def set(self, data_set) -> None:
        data = self._data
        data_type = self.data_type

        if data_type == E_SharedSaveType.Queue:
            data.put(data_set)

        elif data_type == E_SharedSaveType.Tensor:
            data[:] = data_set[:]

        elif data_type == E_SharedSaveType.SharedArray_Int or \
                data_type == E_SharedSaveType.SharedArray_Float:
            data_set = np.asarray(data_set)
            data_set = data_set.flatten()
            data[:] = data_set[:]

        elif data_type == E_SharedSaveType.SharedValue_Int or \
                data_type == E_SharedSaveType.SharedValue_Float:
            data.value = data_set

    def get(self) -> any:
        data_type = self.data_type
        data = self._data

        if data_type == E_SharedSaveType.Queue:
            if data.empty():
                return None
            else:
                return data.get(block=False)

        elif data_type == E_SharedSaveType.Tensor:
            return data

        elif data_type == E_SharedSaveType.SharedArray_Int or \
                data_type == E_SharedSaveType.SharedArray_Float:
            data = data[:]
            data = np.asarray(data)
            data = data.reshape(self.data_shape)
            return data

        elif data_type == E_SharedSaveType.SharedValue_Int or \
                data_type == E_SharedSaveType.SharedValue_Float:
            return data.value

        else:
            raise ValueError

    def clear(self) -> None:
        data_type = self.data_type
        data = self._data

        if data_type == E_SharedSaveType.Queue:
            data: mp.Queue
            while data.qsize() > 0:
                try:
                    data.get()
                except RuntimeError:  # CUDA error: invalid device context
                    pass
            data.close()
        else:
            del data

        torch.cuda.empty_cache()

    def size(self) -> int:
        data_type = self.data_type
        data = self._data

        if data_type == E_SharedSaveType.Queue:
            return data.qsize()
        else:
            return 1
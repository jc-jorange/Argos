import multiprocessing as mp
from enum import Enum, unique
from yacs.config import CfgNode as CN
import numpy as np
import torch
import math
from collections import Iterable


@unique
class E_PipelineSharedDataName(Enum):
    ImageData = 1
    FrameID = 2
    ImageOriginShape = 3
    ImageTimestamp = 4
    CameraTransform = 5
    TransformTimestamp = 6
    CamIntrinsicPara = 7


@unique
class E_SharedSaveType(Enum):
    Queue = 1
    Tensor = 2
    SharedArray_Int = 3
    SharedArray_Float = 4
    SharedValue_Int = 5
    SharedValue_Float = 6


@unique
class E_OutputPortDataType(Enum):
    Default = 1
    CameraTrack = 2


dict_OutputPortDataType = {
    E_OutputPortDataType.Default.name: any,
    E_OutputPortDataType.CameraTrack.name: (int, int, np.ndarray),
}

format_SharedDataInfo = (E_SharedSaveType, tuple, int)


class Struc_ConsumerOutputPort:
    def __init__(self,
                 opt,
                 output_type: E_SharedSaveType,
                 data_type: E_OutputPortDataType,
                 data_shape: tuple
                 ):
        self.opt = opt

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
            result = self.output.get()

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

        if self.output_type == E_SharedSaveType.Queue:
            self.output: mp.Queue
            self.output.put(data)
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
            with self.output.mutex:
                self.output.queue.clear()
        else:
            del self.output

        torch.cuda.empty_cache()


class Struc_SharedData:
    _data = None
    data_type = None
    data_shape = ()
    device = 'cpu'

    def _generate_output_value(self, output_format: format_SharedDataInfo) -> any:
        data_type = output_format[0]
        data_type = data_type if isinstance(data_type, E_SharedSaveType) else E_SharedSaveType[data_type]
        self.data_type = data_type
        data_shape = output_format[1]
        self.data_shape = tuple(data_shape)
        default_value = output_format[2]

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
                 output_format: format_SharedDataInfo):
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
            with data.mutex:
                data.queue.clear()
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


from .process import factory_process_all


class SharedDataHub:
    def __init__(self,
                 device: str,
                 pipeline_cfg: CN):

        self.dict_shared_data = {}
        for pipeline_name, pipeline_branch in pipeline_cfg.items():
            tmp_shared_data_dict = {}
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                if pipeline_leaf and (pipeline_branch_name in factory_process_all.keys()):
                    for pipeline_leaf_name in pipeline_leaf.keys():
                        for shared_data_name, shared_data_info in factory_process_all[pipeline_branch_name] \
                                [pipeline_leaf_name].shared_data.items():
                            tmp_shared_data_dict.update({shared_data_name: Struc_SharedData(device, shared_data_info)})
            self.dict_shared_data.update({pipeline_name: tmp_shared_data_dict})

        self.dict_consumer_port = {
            pipeline_name: [] for pipeline_name, pipeline_branch in pipeline_cfg.items()
        }

        self.dict_process_results_dir = {}
        for pipeline_name, pipeline_branch in pipeline_cfg.items():
            tmp_dict_branch = {}
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                tmp_dict_leaf = {}
                if pipeline_leaf:
                    for pipeline_leaf_name in pipeline_leaf.keys():
                        tmp_dict_leaf[pipeline_leaf_name] = ''
                tmp_dict_branch.update({pipeline_branch_name: tmp_dict_leaf})
            self.dict_process_results_dir.update({pipeline_name: tmp_dict_branch})

        self.dict_bLoadingFlag = {
            pipeline_name: mp.Value('b', 1) for pipeline_name in pipeline_cfg.keys()
        }

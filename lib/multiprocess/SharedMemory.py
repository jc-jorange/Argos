import multiprocessing as mp
from enum import Enum, unique
import numpy as np
import torch


@unique
class E_SharedSaveType(Enum):
    Queue = 1
    Tensor = 2
    SharedArray_Int = 3
    SharedArray_Float = 4
    SharedValue_Int = 5
    SharedValue_Float = 6


@unique
class E_ProducerOutputName_Indi(Enum):
    FrameID = 1
    ImageOriginShape = 2
    ImageData = 3
    bInputLoading = 4
    CameraTransform = 5


@unique
class E_ProducerOutputName_Global(Enum):
    bInputLoading = 1


@unique
class E_ProducerOutputName_Global_PassThrough(Enum):
    PredictAll = 1
    CameraTransformAll = 2


format_ProducerOutput = (E_SharedSaveType, tuple, int)
dict_ProducerOutput_Indi = {
    E_ProducerOutputName_Indi.FrameID: (E_SharedSaveType.SharedValue_Int, (1,), 0),
    E_ProducerOutputName_Indi.ImageOriginShape: (E_SharedSaveType.SharedArray_Int, (3,), 0),
    E_ProducerOutputName_Indi.ImageData: (E_SharedSaveType.Queue, (1,), 0),
    E_ProducerOutputName_Indi.bInputLoading: (E_SharedSaveType.SharedValue_Int, (1,), 1),
    E_ProducerOutputName_Indi.CameraTransform: (E_SharedSaveType.SharedArray_Float, (4, 4), 0),
}

dict_ProducerOutput_Global = {
    E_ProducerOutputName_Global.bInputLoading: (E_SharedSaveType.SharedValue_Int, (1,), 1),
}

dict_ProducerOutput_Global_PassThrough = {
    E_ProducerOutputName_Global_PassThrough.PredictAll: (E_SharedSaveType.Queue, (1,), 0),
    E_ProducerOutputName_Global_PassThrough.CameraTransformAll: (E_SharedSaveType.Queue, (1,), 0),
}


class ProducerHub:
    def __init__(self, opt):
        self.opt = opt
        self.output = None

    def generate_output_value(self, output_format: format_ProducerOutput) -> any:
        data_type: E_SharedSaveType = output_format[0]
        data_shape: tuple = output_format[1]
        default_value: int = output_format[2]

        if data_type == E_SharedSaveType.Queue:
            output_value = mp.Queue()

        elif data_type == E_SharedSaveType.Tensor:
            if default_value:
                output_value = torch.ones(data_shape, dtype=torch.float, device=self.opt.device)
            else:
                output_value = torch.empty(data_shape, dtype=torch.float, device=self.opt.device)
            output_value.share_memory_()

        elif data_type == E_SharedSaveType.SharedArray_Int:
            total_size = sum(data_shape)
            default_value_list = [default_value] * total_size
            output_value = mp.Array('i', total_size)
            output_value[:] = default_value_list[:]

        elif data_type == E_SharedSaveType.SharedArray_Float:
            total_size = sum(data_shape)
            default_value_list = [default_value] * total_size
            output_value = mp.Array('f', total_size)
            output_value[:] = default_value_list[:]

        elif data_type == E_SharedSaveType.SharedValue_Int:
            output_value = mp.Value('i', default_value)

        elif data_type == E_SharedSaveType.SharedValue_Float:
            output_value = mp.Value('f', default_value)

        else:
            raise ValueError(f'Wrong producer output type as {data_type}')

        return output_value


class ProducerHub_Indi(ProducerHub):
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super(ProducerHub_Indi, self).__init__(*args, **kwargs)

        if self.opt.realtime:
            dict_ProducerOutput_Indi[E_ProducerOutputName_Indi.ImageData] = \
                (E_SharedSaveType.Tensor, self.opt.net_input_shape, 0)
        self.output = {}
        for k, v in dict_ProducerOutput_Indi.items():
            self.output[k] = self.generate_output_value(v)


class ProducerHub_Global(ProducerHub):
    def __init__(self,
                 indi_hub: dict,
                 *args,
                 **kwargs
                 ):
        super(ProducerHub_Global, self).__init__(*args, **kwargs)

        self.indi_hub_dict = indi_hub

        self.output_passthrough = {}
        for k_i, v_i in self.indi_hub_dict.items():
            self.output_passthrough[k_i] = {}
            for k_g, v_g in dict_ProducerOutput_Global_PassThrough.items():
                self.output_passthrough[k_i][k_g] = self.generate_output_value(v_g)

        self.output = {}
        for k, v in dict_ProducerOutput_Global.items():
            self.output[k] = self.generate_output_value(v)


class ConsumerOutputPort:
    def __init__(self,
                 opt,
                 output_type: E_SharedSaveType,
                 data_shape: tuple
                 ):
        self.opt = opt

        self._output_type = output_type
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

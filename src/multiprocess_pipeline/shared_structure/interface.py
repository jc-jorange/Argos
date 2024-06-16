from enum import Enum, unique


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


@unique
class E_SharedDataFormat(Enum):
    data_type = 1
    data_shape = 2
    data_value = 3


import numpy as np


dict_OutputPortDataType = {
    E_OutputPortDataType.Default.name: any,
    E_OutputPortDataType.CameraTrack.name: (int, int, np.ndarray),
}

dict_SharedDataInfoFormat = {
    E_SharedDataFormat.data_type.name: E_SharedSaveType,
    E_SharedDataFormat.data_shape.name: tuple,
    E_SharedDataFormat.data_value.name: any,
}

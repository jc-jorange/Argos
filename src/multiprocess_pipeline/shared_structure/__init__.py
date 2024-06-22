from enum import Enum, unique

from .ouput_port import *
from .shared_data import *
from .data_hub import *


@unique
class E_PipelineSharedDataName(Enum):
    ImageData = 1
    FrameID = 2
    ImageOriginShape = 3
    ImageTimestamp = 4
    CameraTransform = 5
    TransformTimestamp = 6
    CamIntrinsicPara = 7

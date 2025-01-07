from enum import Enum, unique

from .ouput_port import *
from .shared_data import *
from .data_hub import *


@unique
class E_PipelineSharedDataName(Enum):
    Image = 1

    TrackedImageInfo = 3

    CameraTransform = 5
    TransformTimestamp = 6
    CamIntrinsicPara = 7

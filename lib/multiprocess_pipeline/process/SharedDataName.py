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

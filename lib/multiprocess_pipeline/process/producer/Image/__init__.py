from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType
from lib.multiprocess_pipeline.process import ProducerProcess
from lib.multiprocess_pipeline.process.SharedDataName import E_PipelineSharedDataName


class ImageProcess_Master(ProducerProcess):
    shared_data = {
        E_PipelineSharedDataName.FrameID.name: (E_SharedSaveType.Queue, (1,), 0),
        E_PipelineSharedDataName.ImageOriginShape.name: (E_SharedSaveType.SharedArray_Int, (3,), 0),
        E_PipelineSharedDataName.ImageData.name: (E_SharedSaveType.Queue, (1,), 0),
        E_PipelineSharedDataName.ImageTimestamp.name: (E_SharedSaveType.Queue, (1,), 0),
    }

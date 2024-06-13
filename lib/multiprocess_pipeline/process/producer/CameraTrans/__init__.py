from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType, E_PipelineSharedDataName
from lib.multiprocess_pipeline.process import ProducerProcess


class CameraTransProcess_Master(ProducerProcess):
    shared_data = {
        E_PipelineSharedDataName.CameraTransform.name: (E_SharedSaveType.Queue, (1,), 0),
        E_PipelineSharedDataName.TransformTimestamp.name: (E_SharedSaveType.Queue, (1,), 0),
    }

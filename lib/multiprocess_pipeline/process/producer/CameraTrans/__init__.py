from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType, E_PipelineSharedDataName, E_SharedDataFormat
from lib.multiprocess_pipeline.process import ProducerProcess


class CameraTransProcess_Master(ProducerProcess):
    shared_data = {
        E_PipelineSharedDataName.CameraTransform.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.TransformTimestamp.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
    }

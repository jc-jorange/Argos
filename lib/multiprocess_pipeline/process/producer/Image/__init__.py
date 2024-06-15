from lib.multiprocess_pipeline.SharedMemory import E_SharedSaveType, E_PipelineSharedDataName, E_SharedDataFormat
from lib.multiprocess_pipeline.process import ProducerProcess


class ImageProcess_Master(ProducerProcess):
    shared_data = {
        E_PipelineSharedDataName.ImageData.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.FrameID.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.ImageOriginShape.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.SharedArray_Int,
            E_SharedDataFormat.data_shape.name: (3,),
            E_SharedDataFormat.data_value.name: 0
        },

        E_PipelineSharedDataName.ImageTimestamp.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
    }

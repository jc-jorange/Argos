from enum import Enum, unique

from lib.multiprocess.individual_process.producer.MP_ImageLoader import ImageLoaderProcess


@unique
class E_Indi_Process_Producer(Enum):
    ImageLoader = 1
    CameraTransformLoader = 2


factory_indi_process_producer = {
    E_Indi_Process_Producer.ImageLoader.name: ImageLoaderProcess,
}

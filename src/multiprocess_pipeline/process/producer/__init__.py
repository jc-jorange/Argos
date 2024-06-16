from enum import Enum, unique

from .Image.ImageLoader import ImageLoaderProcess
from .CameraTrans.CameraTransLoader import CameraTransLoaderProcess


@unique
class E_Process_Producer(Enum):
    ImageLoader = 1
    CameraTransLoader = 2


factory_process_producer = {
    E_Process_Producer.ImageLoader.name: ImageLoaderProcess,
    E_Process_Producer.CameraTransLoader.name: CameraTransLoaderProcess,
}

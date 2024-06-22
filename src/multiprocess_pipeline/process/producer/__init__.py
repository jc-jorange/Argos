from enum import Enum, unique

from ._masterclass import *

from .ImageLoader import ImageLoaderProcess
from .CameraTransLoader import CameraTransLoaderProcess


@unique
class E_Process_Producer(Enum):
    ImageLoader = 1
    CameraTransLoader = 2


factory_process_producer = {
    E_Process_Producer.ImageLoader.name: ImageLoaderProcess,
    E_Process_Producer.CameraTransLoader.name: CameraTransLoaderProcess,
}

__all__ = [
    ProducerProcess,
    E_Process_Producer,
    factory_process_producer,
]

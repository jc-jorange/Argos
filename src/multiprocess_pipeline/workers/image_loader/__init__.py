from enum import Enum, unique

from ._masterclass import *

from .image import ImageDataLoader
from .video import VideoDataLoader
from .address import AddressImageLoader


@unique
class E_ImageLoaderName(Enum):
    Image = 1
    Video = 2
    Address = 3


factory_image_loader = {
    E_ImageLoaderName.Image.name: ImageDataLoader,
    E_ImageLoaderName.Video.name: VideoDataLoader,
    E_ImageLoaderName.Address.name: AddressImageLoader,
}

__all__ = [
    BaseImageLoader,
    E_ImageLoaderName,
    factory_image_loader
]

from enum import Enum, unique

from ._masterclass import BasePost

from .result_writer import TextResultWriter, ImageResultWriter, VideoResultWriter


@unique
class E_PostFactory(Enum):
    TextWriter = 1
    ImageWriter = 2
    VideoWriter = 3


factory_post = {
    E_PostFactory.TextWriter.name: TextResultWriter,
    E_PostFactory.ImageWriter.name: ImageResultWriter,
    E_PostFactory.VideoWriter.name: VideoResultWriter,
}

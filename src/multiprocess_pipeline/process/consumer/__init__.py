from enum import Enum, unique

from ._masterclass import *

from .Track import TrackerProcess
from .PathPredict import PathPredictProcess
from .MultiCameraIdMatch import MultiCameraIdMatchProcess
from .DataSend import DataSendProcess
from .DataSmooth import DataSmoothProcess


@unique
class E_Process_Consumer(Enum):
    Track = 1
    PathPredict = 2
    MultiCameraIdMatch = 3
    DataSend = 4
    DataSmooth = 5


factory_process_consumer = {
    E_Process_Consumer.Track.name: TrackerProcess,
    E_Process_Consumer.PathPredict.name: PathPredictProcess,
    E_Process_Consumer.MultiCameraIdMatch.name: MultiCameraIdMatchProcess,
    E_Process_Consumer.DataSend.name: DataSendProcess,
    E_Process_Consumer.DataSmooth.name: DataSmoothProcess,
}

__all__ = [
    ConsumerProcess,
    E_Process_Consumer,
    factory_process_consumer
]

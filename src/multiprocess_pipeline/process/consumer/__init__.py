from enum import Enum, unique

from ._masterclass import *

from .Track import TrackerProcess
from .PathPredict import PathPredictProcess
from .MultiCameraIdMatch import MultiCameraIdMatchProcess
from .DataSend import DataSendProcess


@unique
class E_Process_Consumer(Enum):
    Track = 1
    PathPredict = 2
    MultiCameraIdMatch = 3
    DataSend = 4


factory_process_consumer = {
    E_Process_Consumer.Track.name: TrackerProcess,
    E_Process_Consumer.PathPredict.name: PathPredictProcess,
    E_Process_Consumer.MultiCameraIdMatch.name: MultiCameraIdMatchProcess,
    E_Process_Consumer.DataSend.name: DataSendProcess,
}

__all__ = [
    ConsumerProcess,
    E_Process_Consumer,
    factory_process_consumer
]

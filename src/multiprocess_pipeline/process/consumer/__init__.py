from enum import Enum, unique

from .Tracker import TrackerProcess
from .PathPredict import PathPredictProcess
from .MultiCameraIdMatch import MultiCameraIdMatchProcess


@unique
class E_Process_Consumer(Enum):
    Track = 1
    PathPredict = 2
    MultiCameraIdMatch = 3


factory_process_consumer = {
    E_Process_Consumer.Track.name: TrackerProcess,
    E_Process_Consumer.PathPredict.name: PathPredictProcess,
    E_Process_Consumer.MultiCameraIdMatch.name: MultiCameraIdMatchProcess,
}

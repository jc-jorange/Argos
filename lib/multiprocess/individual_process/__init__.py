from enum import Enum, unique

from .MP_ImageLoader import ImageLoaderProcess
from .MP_Tracker import TrackerProcess
from .MP_PathPredict import PathPredictProcess
from .MP_IndiPost import IndividualPostProcess


@unique
class E_Indi_Process(Enum):
    ImageLoader = 1
    Tracker = 2
    Predictor = 3
    IndiPost = 4


factory_indi_process = {
    E_Indi_Process.ImageLoader.name: ImageLoaderProcess,
    E_Indi_Process.Tracker.name: TrackerProcess,
    E_Indi_Process.Predictor.name: PathPredictProcess,
    E_Indi_Process.IndiPost.name: IndividualPostProcess,
}

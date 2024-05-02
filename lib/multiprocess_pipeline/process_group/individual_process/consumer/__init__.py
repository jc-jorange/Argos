from enum import Enum, unique

from lib.multiprocess_pipeline.process_group.individual_process.consumer.MP_Tracker import TrackerProcess
from lib.multiprocess_pipeline.process_group.individual_process.consumer.MP_PathPredict import PathPredictProcess


@unique
class E_Indi_Process_Consumer(Enum):
    Tracker = 1
    Predictor = 2


factory_indi_process_consumer = {
    E_Indi_Process_Consumer.Tracker.name: TrackerProcess,
    E_Indi_Process_Consumer.Predictor.name: PathPredictProcess,
}
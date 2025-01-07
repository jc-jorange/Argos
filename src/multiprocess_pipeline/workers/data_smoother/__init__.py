from enum import Enum, unique

from .SimpleMovingAverage import SimpleMovingAverage
from .SavitzkyGolay import SavitzkyGolay
from .KalmanFilter import KalmanFilter


@unique
class E_DataSmootherName(Enum):
    SimpleMovingAverage = 1
    SavitzkyGolay = 2
    KalmanFilter = 3


factory_data_smoother = {
    E_DataSmootherName.SimpleMovingAverage.name: SimpleMovingAverage,
    E_DataSmootherName.SavitzkyGolay.name: SavitzkyGolay,
    E_DataSmootherName.KalmanFilter.name: KalmanFilter,
}

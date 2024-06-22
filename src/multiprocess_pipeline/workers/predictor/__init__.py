from enum import Enum, unique

from .spline.linear import LinearSpline
from .spline.hermite import HermiteSpline


@unique
class E_PredictorName(Enum):
    LinearSpline = 1
    HermiteSpline = 2


factory_predictor = {
    E_PredictorName.LinearSpline.name: LinearSpline,
    E_PredictorName.HermiteSpline.name: HermiteSpline,
}

from enum import Enum, unique

from ._masterclass import *

from .ResultsVisualize.ResultsVisualize_indi import IndiResultsVisualizeProcess
from .ResultsVisualize.ResultsVisualize_global import GloResultsVisualizeProcess


@unique
class E_Process_Post(Enum):
    IndiResultsVisual = 1
    GlobalResultsVisual = 2


factory_process_post = {
    E_Process_Post.IndiResultsVisual.name: IndiResultsVisualizeProcess,
    E_Process_Post.GlobalResultsVisual.name: GloResultsVisualizeProcess,
}

__all__ = [
    PostProcess,
    E_Process_Post,
    factory_process_post,
]

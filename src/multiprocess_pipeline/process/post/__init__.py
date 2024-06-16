from enum import Enum, unique

from .ResultsVisualize.IndiPost import IndiResultsVisualizeProcess
from .ResultsVisualize.GlobalPost import GloResultsVisualizeProcess


@unique
class E_Process_Post(Enum):
    IndiResultsVisual = 1
    GlobalResultsVisual = 2


factory_process_post = {
    E_Process_Post.IndiResultsVisual.name: IndiResultsVisualizeProcess,
    E_Process_Post.GlobalResultsVisual.name: GloResultsVisualizeProcess,
}

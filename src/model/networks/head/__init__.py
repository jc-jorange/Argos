from enum import Enum, unique

from ._masterclass import *

from .FairMOT import FairMOT


@unique
class E_HeadName(Enum):
    FairMOT = 1


head_factory_ = {
    E_HeadName.FairMOT.name: FairMOT.FairMOT,
}

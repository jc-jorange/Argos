from enum import Enum, unique

from ._masterclass import *

from .FPN import fpn
from .Ghost_PAN import ghost_pan
from .PAN import pan
from .TAN import tan
from .DLA_Fusion import dla_fusion


@unique
class E_NeckName(Enum):
    FPN = 1
    Ghost_PAN = 2
    PAN = 3
    TAN = 4
    DLA_Fusion = 5


neck_factory_ = {
    E_NeckName.FPN.name: fpn.FPN,
    E_NeckName.Ghost_PAN.name: ghost_pan.GhostPAN,
    E_NeckName.PAN.name: pan.PAN,
    E_NeckName.TAN.name: tan.TAN,
    E_NeckName.DLA_Fusion.name: dla_fusion.DLA_Fusion,
}

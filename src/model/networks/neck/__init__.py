import torch.nn as nn
from src.utils.utils import *


class BaseModel_neck(nn.Module):
    def __init__(self, input_dim=(64, 128, 256), **kwargs):
        self.input_dim = input_dim
        super(BaseModel_neck, self).__init__()


from .FPN import fpn
from .Ghost_PAN import ghost_pan
from .PAN import pan
from .TAN import tan
from .DLA_Fusion import dla_fusion

neck_factory_ = {
    model_dir_name(fpn.__file__): fpn.FPN,
    model_dir_name(ghost_pan.__file__): ghost_pan.GhostPAN,
    model_dir_name(pan.__file__): pan.PAN,
    model_dir_name(fpn.__file__): tan.TAN,
    model_dir_name(dla_fusion.__file__): dla_fusion.DLA_Fusion,
}

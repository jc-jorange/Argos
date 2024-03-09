import torch
import torch.nn as nn

class BaseModel_loss(nn.Module):
    def __init__(self, opt, cfg, model_info):
        super(BaseModel_loss, self).__init__()
        self.opt = opt
        self.cfg = cfg
        self.model_info = model_info

    def forward(self, outputs, batch) -> (torch.Tensor, dict):
        pass


from lib.model.networks.loss.losses import FocalLoss
from lib.model.networks.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.model.utils import _sigmoid


_loss_factory = {
    'MSELoss': torch.nn.MSELoss(),
    'FocalLoss': FocalLoss(),
    'RegL1Loss': RegL1Loss(),
    'RegLoss': RegLoss(),
    'L1Loss': torch.nn.L1Loss(reduction='sum'),
    'NormRegL1Loss': NormRegL1Loss(),
    'RegWeightedL1Loss': RegWeightedL1Loss(),
}
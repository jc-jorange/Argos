import torch

from ._masterclass import *

from src.model.networks.loss.losses import FocalLoss
from src.model.networks.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from src.model.utils import _sigmoid


_loss_factory = {
    'MSELoss': torch.nn.MSELoss(),
    'FocalLoss': FocalLoss(),
    'RegL1Loss': RegL1Loss(),
    'RegLoss': RegLoss(),
    'L1Loss': torch.nn.L1Loss(reduction='sum'),
    'NormRegL1Loss': NormRegL1Loss(),
    'RegWeightedL1Loss': RegWeightedL1Loss(),
}

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

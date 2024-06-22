import torch.nn as nn


class BaseModel_backbone_with_neck(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel_backbone_with_neck, self).__init__()

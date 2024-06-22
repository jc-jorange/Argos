import torch.nn as nn


class BaseModel_backbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel_backbone, self).__init__()

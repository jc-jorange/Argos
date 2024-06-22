import torch.nn as nn


class BaseModel_neck(nn.Module):
    def __init__(self, input_dim=(64, 128, 256), **kwargs):
        self.input_dim = input_dim
        super(BaseModel_neck, self).__init__()

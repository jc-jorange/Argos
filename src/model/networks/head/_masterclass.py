import torch.nn as nn


class BaseModel_head(nn.Module):
    def __init__(self, loss_class, loss_cfg, num_max_ids=128, num_max_classes=10, input_dim=128):
        super(BaseModel_head, self).__init__()
        self.num_max_classes = num_max_classes
        self.num_max_ids = num_max_ids
        self.input_dim = input_dim
        self.loss_class = loss_class
        self.loss_cfg = loss_cfg

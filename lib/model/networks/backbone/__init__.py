import torch.nn as nn
from ..utils import *


class BaseModel_backbone(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel_backbone, self).__init__()


from .CSP_Custom import custom_csp
from .EfficientNet_Lite import efficientnet_lite
from .GhostNet import ghostnet
from .MobileNet_V2 import mobilenetv2
from .RepVGG import repvgg
from .ResNet import resnet
from .ShuffleNetV2 import shufflenetv2
from .ParNet import ParNet
from .ParNet_Mod import ParNet_Mod
from .DLA import dla
from .EfficientNet import EfficientNet

backbone_factory_ = {
    model_dir_name(custom_csp.__file__): custom_csp.CustomCspNet,
    model_dir_name(efficientnet_lite.__file__): efficientnet_lite.EfficientNetLite,
    model_dir_name(ghostnet.__file__): ghostnet.GhostNet,
    model_dir_name(mobilenetv2.__file__): mobilenetv2.MobileNetV2,
    model_dir_name(repvgg.__file__): repvgg.RepVGG,
    model_dir_name(resnet.__file__): resnet.ResNet,
    model_dir_name(shufflenetv2.__file__): shufflenetv2.ShuffleNetV2,
    model_dir_name(ParNet.__file__): ParNet.ParNet,
    model_dir_name(ParNet_Mod.__file__): ParNet_Mod.ParNet_Mod,
    model_dir_name(dla.__file__): dla.DLA,
    model_dir_name(EfficientNet.__file__): EfficientNet.EfficientNet.from_pretrained,
}

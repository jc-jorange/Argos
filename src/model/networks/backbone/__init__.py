from enum import Enum, unique

from ._masterclass import BaseModel_backbone

from .CSP_Custom import custom_csp
from .GhostNet import ghostnet
from .MobileNet_V2 import mobilenetv2
from .RepVGG import repvgg
from .ResNet import resnet
from .ShuffleNetV2 import shufflenetv2
from .ParNet import ParNet
from .ParNet_Mod import ParNet_Mod
from .DLA import dla
from .EfficientNet import EfficientNet


@unique
class E_BackboneName(Enum):
    CSP_Custom = 1
    EfficientNet_Lite = 2
    GhostNet = 3
    MobileNet_V2 = 4
    RepVGG = 5
    ResNet = 6
    ShuffleNetV2 = 7
    ParNet = 8
    ParNet_Mod = 9
    DLA = 10
    EfficientNet = 11


backbone_factory_ = {
    E_BackboneName.CSP_Custom.name: custom_csp.CustomCspNet,
    E_BackboneName.GhostNet.name: ghostnet.GhostNet,
    E_BackboneName.MobileNet_V2.name: mobilenetv2.MobileNetV2,
    E_BackboneName.RepVGG.name: repvgg.RepVGG,
    E_BackboneName.ResNet.name: resnet.ResNet,
    E_BackboneName.ShuffleNetV2.name: shufflenetv2.ShuffleNetV2,
    E_BackboneName.ParNet.name: ParNet.ParNet,
    E_BackboneName.ParNet_Mod.name: ParNet_Mod.ParNet_Mod,
    E_BackboneName.DLA.name: dla.DLA,
    E_BackboneName.EfficientNet.name: EfficientNet.EfficientNet.from_pretrained,
}

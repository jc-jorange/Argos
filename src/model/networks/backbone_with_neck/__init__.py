from enum import Enum, unique

from ._masterclass import *

from .EDANet import EDANet
from .CGNet import CGNet
from .CSPDarkNet import csp_darknet
from .DLAv0 import dlav0
from .Fast_SCNN import fast_scnn
from .ParNet import parnet
from .UNetX import UNetX
from .DLA_DCN import pose_dla_dcn
from .HRNet import pose_hrnet
from .RepVGG_Plus import repvggplus
from .ResNet_DCN import resnet_dcn

@unique
class E_BackboneWithNeckName(Enum):
    EDANet = 1
    CGNet = 2
    CSPDarkNet = 3
    DLAv0 = 4
    Fast_SCNN = 5
    ParNet = 6
    UNetX = 7
    DLA_DCN = 8
    HRNet = 9
    RepVGG_Plus = 10
    ResNet_DCN = 11


backbone_with_neck_factory_ = {
    E_BackboneWithNeckName.EDANet.name: EDANet.EDANet,
    E_BackboneWithNeckName.CGNet.name: CGNet.Context_Guided_Network,
    E_BackboneWithNeckName.CSPDarkNet.name: csp_darknet.CSPDarkNet,
    E_BackboneWithNeckName.DLAv0.name: dlav0.DLASeg,
    E_BackboneWithNeckName.Fast_SCNN.name: fast_scnn.FastSCNN,
    E_BackboneWithNeckName.ParNet.name: parnet.ParNet,
    E_BackboneWithNeckName.UNetX.name: UNetX.UNetX,
    E_BackboneWithNeckName.DLA_DCN.name: pose_dla_dcn.DLASeg,
    E_BackboneWithNeckName.HRNet.name: pose_hrnet.PoseHighResolutionNet,
    E_BackboneWithNeckName.RepVGG_Plus.name: repvggplus.RepVGGplus,
    E_BackboneWithNeckName.ResNet_DCN: resnet_dcn.PoseResNet,
}

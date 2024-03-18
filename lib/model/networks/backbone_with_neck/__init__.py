import torch.nn as nn
from lib.utils.utils import *


class BaseModel_backbone_with_neck(nn.Module):
    def __init__(self, **kwargs):
        super(BaseModel_backbone_with_neck, self).__init__()


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

backbone_with_neck_factory_ = {
    model_dir_name(EDANet.__file__): EDANet.EDANet,
    model_dir_name(CGNet.__file__): CGNet.Context_Guided_Network,
    model_dir_name(csp_darknet.__file__): csp_darknet.CSPDarkNet,
    model_dir_name(dlav0.__file__): dlav0.DLASeg,
    model_dir_name(fast_scnn.__file__): fast_scnn.FastSCNN,
    model_dir_name(parnet.__file__): parnet.ParNet,
    model_dir_name(UNetX.__file__): UNetX.UNetX,
    model_dir_name(pose_dla_dcn.__file__): pose_dla_dcn.DLASeg,
    model_dir_name(pose_hrnet.__file__): pose_hrnet.PoseHighResolutionNet,
    model_dir_name(repvggplus.__file__): repvggplus.RepVGGplus,
    model_dir_name(resnet_dcn.__file__): resnet_dcn.PoseResNet,
}

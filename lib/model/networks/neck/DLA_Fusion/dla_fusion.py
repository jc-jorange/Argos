import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import BaseModel_neck

import numpy as np
import math
from dcn_v2 import DCN

from lib.model.networks.module.ghost_module import GhostBottleneck
from lib.model.networks.module.Squeeze_and_Excitation import SELayer
from lib.model.networks.module.Convolutional_Block_Attention_Module import CBAM

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

class DeformConv(nn.Module):
    def __init__(self, chi, cho):
        super(DeformConv, self).__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(cho, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.conv = DCN(chi, cho, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f, use_ghost_module=False, use_se=False, use_cbam=False):
        super(IDAUp, self).__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o) if not use_ghost_module else GhostBottleneck(c, c, o)
            node = DeformConv(o, o) if not use_ghost_module else GhostBottleneck(o, o, o)

            if use_se:
                se = SELayer(o)
                setattr(self, 'se_' + str(i), se)
            if use_cbam:
                cbam = CBAM(o)
                setattr(self, 'cbam_' + str(i), cbam)

            up = nn.ConvTranspose2d(o, o, f * 2, stride=f,
                                    padding=f // 2, output_padding=0,
                                    groups=o, bias=False)
            fill_up_weights(up)

            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)
            setattr(self, 'node_' + str(i), node)

    def forward(self, layers, startp, endp):
        for i in range(startp+1, endp):
            upsample = getattr(self, 'up_' + str(i - startp))
            project = getattr(self, 'proj_' + str(i - startp))
            se_layer = None
            cbam_layer = None
            try:
                se_layer = getattr(self, 'se_' + str(i - startp))
                cbam_layer = getattr(self, 'cbam_' + str(i - startp))
            except:
                pass
            p = project(layers[i])
            layers[i] = upsample(p)
            if se_layer:
                layers[i] = se_layer(layers[i])
            if cbam_layer:
                layers[i] = cbam_layer(layers[i])
            node = getattr(self, 'node_' + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])


class DLAUp(nn.Module):
    def __init__(self, start_p, channels, scales, in_channels=None, use_ghost_module=False, use_se=False, use_cbam=False):
        """
        :param startp:
        :param channels:
        :param scales:
        :param in_channels:
        """
        super(DLAUp, self).__init__()

        self.start_p = start_p

        if in_channels is None:
            in_channels = channels

        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)

        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(channels[j], in_channels[j:],
                          scales[j:] // scales[j], use_ghost_module=use_ghost_module, use_se=use_se, use_cbam=use_cbam))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.start_p - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])

        return out
#
# class SE_layers(nn.Module):
#     def __init__(self, channels):
#         super(SE_layers, self).__init__()
#         self.channels = channels
#         for i in range(len(channels)):
#             setattr(self, 'se_{}'.format(i),
#                     SELayer(channels[i]))
#
#     def forward(self, x, startp, endp):
#         for i in range(startp, endp+1):
#             se = getattr(self, 'se_' + str(i - startp))
#             x[i] = se(x[i])
#
#         return x
#
#
# class CBAM_layers(nn.Module):
#     def __init__(self, channels):
#         super(CBAM_layers, self).__init__()
#         self.channels = channels
#         for i in range(len(channels)):
#             setattr(self, 'cbam_{}'.format(i),
#                     CBAM(channels[i]))
#
#     def forward(self, x, startp, endp):
#         for i in range(startp, endp+1):
#             cbam = getattr(self, 'cbam_' + str(i - startp))
#             x[i] = cbam(x[i])
#
#         return x


# class Interpolate(nn.Module):
#     def __init__(self, scale, mode):
#         super(Interpolate, self).__init__()
#         self.scale = scale
#         self.mode = mode
#
#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
#         return x


class DLA_Fusion(BaseModel_neck):
    def __init__(self,
                 last_level,
                 use_ghost_module=False,
                 use_se=False,
                 use_cbam=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.first_level = 2
        self.last_level = last_level
        channels = [16, 32] + self.input_dim if len(self.input_dim) < last_level else self.input_dim
        self.channels = channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.scales = scales

        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales,
                            use_ghost_module=use_ghost_module, use_se=use_se, use_cbam=use_cbam)

        self.ida_up = IDAUp(channels[self.first_level], channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)],
                            use_ghost_module=use_ghost_module, use_se=use_se, use_cbam=use_cbam)

    def forward(self, x):
        t_s = time.perf_counter()
        x = list(x)
        # if len(x) < self.last_level+1:
        #     for i in range((self.last_level+1)-len(x)):
        #         x.insert(0, None)
        #
        # if self.se_layer:
        #     self.se_layer(x, self.first_level, self.last_level)
        #
        # if self.cbam:
        #     self.cbam(x, self.first_level, self.last_level)

        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        t_e = time.perf_counter()
        t = t_e - t_s
        print('DLA_fusion Neck Time: {}'.format(t))
        return [y[-1]]

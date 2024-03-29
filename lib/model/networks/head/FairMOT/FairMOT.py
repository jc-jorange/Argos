from .. import BaseModel_head

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import math
from ...loss import BaseModel_loss
from lib.model.networks.loss.losses import FocalLoss
from lib.model.networks.loss.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from lib.model.utils import _sigmoid, _tranpose_and_gather_feat

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# def test_weight(layers):
#     for m in layers.modules():
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.fill_(0.0)

class FairMOT(BaseModel_head):
    def __init__(self,
                 head_conv=256,
                 reid_dim=128,
                 **kwargs):
        super(FairMOT, self).__init__(loss_class=McMotLoss, **kwargs)

        self.heads = self.set_heads(self.num_max_classes, reid_dim, self.loss_cfg['cat_spec_wh'], self.loss_cfg['reg_offset'])
        for head in self.heads:
            channels = self.heads[head]
            if head_conv > 0:
                head_out = nn.Sequential(nn.Conv2d(self.input_dim[0], head_conv, kernel_size=3, padding=1, bias=True),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(head_conv, channels, kernel_size=1, stride=1, padding=0, bias=True))
                if 'hm' in head:
                    head_out[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(head_out)
            else:
                head_out = nn.Conv2d(self.input_dim[0], channels, kernel_size=1, stride=1, padding=0, bias=True)
                if 'hm' in head:
                    head_out.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(head_out)
            # test_weight(head_out)
            # set each head
            self.__setattr__(head, head_out)

    def forward(self, x: list):
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x[-1])
        return [ret]

    def set_heads(self, num_classes, id_dim, cat_spec_wh, reg_offset):
        # cat_spec_wh: category specific bounding box size.
        # reg_offset: not regress local offset.
        heads = {
            'hm': num_classes,
            'wh': 4 if not cat_spec_wh else 4 * num_classes,
            'id': id_dim
        }
        if reg_offset:
            heads.update({'reg': 2})

        return heads


# loss function
class McMotLoss(BaseModel_loss):
    def __init__(self, **kwargs):
        super(McMotLoss, self).__init__(**kwargs)

        opt = self.opt
        cfg = self.cfg

        self.crit = torch.nn.MSELoss() if cfg['mse_loss'] else FocalLoss()
        self.crit_reg = RegL1Loss() if cfg['reg_loss'] == 'l1' else \
            RegLoss() if cfg['reg_loss'] == 'sl1' else None  # L1 loss or smooth l1 loss
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if cfg['dense_wh'] else \
            NormRegL1Loss() if cfg['norm_wh'] else \
                RegWeightedL1Loss() if cfg['cat_spec_wh'] else self.crit_reg  # box size loss

        # @even: Test additional loss functions for re-id
        # self.circle_loss = CircleLoss(m=0.25, gamma=80)
        # self.ghm_c = GHMC(bins=30)  # GHM_C loss for multi-class classification(For ReID)

        if cfg['id_weight'] > 0:
            self.emb_dim = self.model_info.model.Main.head.heads['id']

            # @even: 用nID_dict取代nID, 用于MCMOT(multi-class multi-object tracking)训练
            self.nID_dict = self.model_info.nID_dict

            # 包含可学习参数的层: 用于Re-ID的全连接层
            # @even: 为每个需要ReID的类别定义一个分类器
            self.classifiers = nn.ModuleDict()  # 使用ModuleList或ModuleDict才可以自动注册参数
            # self.focal_loss_dict = nn.ModuleDict()
            for cls_id, nID in self.nID_dict.items():
                # 选择一: 使用普通的全连接层
                self.classifiers[str(cls_id)] = nn.Linear(self.emb_dim, nID)  # FC layers

                # 选择二: 使用Arc margin全连接层
                # self.classifiers[str(cls_id)] = ArcMarginFc(self.emb_dim, nID, self.opt.device, 0.3)

                # 选择三: 使用Focal loss
                # self.focal_loss_dict[str(cls_id)] = McFocalLoss(nID, self.opt.device)

            # using CE loss to do ReID classification
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
            # self.TriLoss = TripletLoss()

            # @even: 为每个需要ReID的类别定义一个embedding scale
            self.emb_scale_dict = dict()
            for cls_id, nID in self.nID_dict.items():
                self.emb_scale_dict[cls_id] = math.sqrt(2) * math.log(nID - 1)

            # track reid分类的损失缩放系数
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))  # -1.05

        # scale factor of detection loss
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))

    def forward(self, outputs, batch):
        """
        :param outputs:
        :param batch:
        :return:
        """
        opt = self.opt
        cfg = self.cfg

        # Initial loss
        hm_loss, wh_loss, off_loss, reid_loss = 0, 0, 0, 0
        for s in range(opt.num_stacks):
            # ----- Detection loss
            output = outputs[s]
            if not cfg['mse_loss']:
                output['hm'] = _sigmoid(output['hm'])

            # --- heat-map loss
            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks

            # --- box width and height loss
            if cfg['wh_weight'] > 0:
                if cfg['dense_wh']:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                             batch['dense_wh'] * batch['dense_wh_mask']) / mask_weight) \
                               / opt.num_stacks
                else:  # box width and height using L1/Smooth L1 loss
                    wh_loss += self.crit_reg(output['wh'], batch['reg_mask'],
                                             batch['ind'], batch['wh']) / opt.num_stacks

            # --- bbox center offset loss
            if cfg['reg_offset'] and cfg['off_weight'] > 0:  # offset using L1 loss
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            # ----- ReID loss: only process the class requiring ReID
            if cfg['id_weight'] > 0:  # if ReID is needed
                cls_id_map = batch['cls_id_map']

                # 遍历每一个需要ReID的检测类别, 计算ReID损失
                for cls_id, id_num in self.nID_dict.items():
                    inds = torch.where(cls_id_map == cls_id)
                    if inds[0].shape[0] == 0:
                        # print('skip class id', cls_id)
                        continue

                    # --- 取cls_id对应索引处的特征向量
                    cls_id_head = output['id'][inds[0], :, inds[2], inds[3]]
                    cls_id_head = self.emb_scale_dict[cls_id] * F.normalize(cls_id_head)  # n × emb_dim
                    # cls_id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                    # cls_id_head = cls_id_head[batch['reg_mask'] > 0].contiguous()
                    # cls_id_head = self.emb_scale_dict[cls_id] * F.normalize(cls_id_head)

                    # --- 获取target类别
                    cls_id_target = batch['cls_tr_ids'][inds[0], cls_id, inds[2], inds[3]]
                    # cls_id_target = batch['ids'][batch['reg_mask'] > 0]

                    # ---分类结果
                    # normal FC layers
                    cls_id_pred = self.classifiers[str(cls_id)](cls_id_head).contiguous()

                    # 使用Arc margin全连接层
                    # cls_id_pred = self.classifiers[str(cls_id)].forward(cls_id_head, cls_id_target).contiguous()

                    # --- 累加每一个检测类别的ReID loss
                    # 选择一: 使用交叉熵优化ReID
                    # print('\nNum objects:'); print(cls_id_target.nelement())
                    # reid_loss += self.ce_loss(cls_id_pred, cls_id_target) / float(cls_id_target.nelement())
                    reid_loss += self.ce_loss(cls_id_pred, cls_id_target) / len(self.nID_dict)

                    # 选择二: 使用Circle loss优化ReID
                    # reid_loss += self.circle_loss(*convert_label_to_similarity(cls_id_pred, cls_id_target))

                    # 选择三: 使用ce_loss + triplet loss优化ReID
                    # reid_loss += self.ce_loss(cls_id_pred, cls_id_target) + self.TriLoss(cls_id_head, cls_id_target)

                    # 选择三: Focal loss
                    # reid_loss += self.focal_loss_dict[str(cls_id)](cls_id_pred, cls_id_target)

                    # 选择四: 使用GHM loss
                    # target = torch.zeros_like(cls_id_pred)
                    # target.scatter_(1, cls_id_target.view(-1, 1).long(), 1)
                    # label_weight = torch.ones_like(cls_id_pred)
                    # reid_loss += self.ghm_c.forward(cls_id_pred, target, label_weight)

        # loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        det_loss = cfg['hm_weight'] * hm_loss \
                   + cfg['wh_weight'] * wh_loss \
                   + cfg['off_weight'] * off_loss

        if cfg['id_weight'] > 0:
            loss = torch.exp(-self.s_det) * det_loss \
                   + torch.exp(-self.s_id) * reid_loss \
                   + (self.s_det + self.s_id)
        else:
            loss = torch.exp(-self.s_det) * det_loss \
                   + self.s_det

        loss *= 0.5
        # print(wh_loss)
        if cfg['id_weight'] > 0:
            loss_stats = {'loss': loss,
                          'hm_loss': hm_loss,
                          'wh_loss': wh_loss,
                          'off_loss': off_loss,
                          'id_loss': reid_loss}
        else:
            loss_stats = {'loss': loss,
                          'hm_loss': hm_loss,
                          'wh_loss': wh_loss,
                          'off_loss': off_loss}  # only exists det loss

        return loss, loss_stats

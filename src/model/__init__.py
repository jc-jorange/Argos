import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import yaml
from collections import OrderedDict
import multiprocessing as mp

from src.multiprocess_pipeline.utils import get_current_process_name

from src.utils.logger import ALL_LoggerContainer
from src.model.model_config import model_cfg_master, merge_config, check_model_architecture, check_model_cfg_exist
from src.model.model_config import E_arch_position, E_model_general_info, E_model_part_input_info, E_model_part_input_info
from src.model.model_config import model_general_info_default_dict
from .networks.model_part_config import E_model_part_info
from .networks import backbone_with_neck, backbone, neck, head


def load_model(model,
               model_path,
               optimizer=None,
               resume=False,
               lr=None,
               lr_step=None):
    """
    """
    local_logger = ALL_LoggerContainer.get_logger(mp.current_process().name)
    start_lr = lr
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    if 'epoch' in checkpoint.keys():
        start_epoch = checkpoint['epoch']
        local_logger.info('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))

    if 'state_dict' in checkpoint.keys():
        state_dict_ = checkpoint['state_dict']
    else:
        state_dict_ = checkpoint
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                local_logger.warning('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                        k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            local_logger.warning('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            local_logger.warning('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            local_logger.info('Resumed optimizer with start lr', start_lr)
        else:
            local_logger.info('No optimizer parameters in checkpoint.')

    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    """
    :param path:
    :param epoch:
    :param model:
    :param optimizer:
    :return:
    """
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    data = {'epoch': epoch,
            'state_dict': state_dict}

    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()

    torch.save(data, path)


class BaseModel(nn.Module):
    def __init__(
            self,
            opt,
            arch: str,
    ):
        super(BaseModel, self).__init__()
        self.logger = ALL_LoggerContainer.get_logger(get_current_process_name())

        self.input_info = {}

        self.opt = opt

        cfg_path = check_model_cfg_exist(self.opt.arch_cfg_path, arch)
        self.cfg = model_cfg_master
        merge_config(self.cfg, cfg_path)

        self.logger.info('model config: {}'.format(arch))

        self.classes_max_num = self.cfg[E_model_general_info.max_classes_num.name] \
            if self.cfg[E_model_general_info.max_classes_num.name] > model_general_info_default_dict[E_model_general_info.max_classes_num] \
            else model_general_info_default_dict[E_model_general_info.max_classes_num]
        self.objects_max_num = self.cfg[E_model_general_info.max_objects_num.name] \
            if self.cfg[E_model_general_info.max_objects_num.name] > model_general_info_default_dict[E_model_general_info.max_objects_num] \
            else model_general_info_default_dict[E_model_general_info.max_objects_num]

        arch_list = check_model_architecture(self.cfg)
        modules_dict = OrderedDict()
        arch_list.sort(key=lambda x: E_arch_position[x].value, reverse=True)
        for i, part in enumerate(arch_list):
            self.logger.info('creating model part: {}'.format(part))
            module = self.builder(part, self.cfg)
            modules_dict.update({part: module})
            if i+1 < len(arch_list):
                self.get_part_input_info(arch_list[i+1], modules_dict)
        self.Main = nn.Sequential(modules_dict)
        del modules_dict

        self.info_data = S_info_data(self)
        self.logger.info('model general info:')
        for k, v in self.info_data.__dict__.items():
            self.logger.info('{}: {}'.format(k, v))

    def forward(self, x):
        output = self.Main(x)
        return output

    def builder(self, part_name: str, cfg_model):
        model_name = cfg_model[part_name][E_model_part_info.model_name.name]
        model_cfg_path = os.path.join(self.opt.model_part_path, part_name, model_name, 'cfg')

        part_cfg_path = \
            check_model_cfg_exist(model_cfg_path, cfg_model[part_name][E_model_part_info.cfg_name.name])

        with open(part_cfg_path, 'r') as f:
            arg = yaml.safe_load(f)

        self.logger.info('model part {} config: '.format(part_name))
        for k, v in arg.items():
            self.logger.info('{}: {}'.format(k, v))

        if part_name == E_arch_position.head.name:  # Head
            return head.head_factory_[model_name](
                num_max_classes=self.classes_max_num,
                num_max_ids=self.objects_max_num,
                input_dim=self.input_info[E_arch_position.head.name][E_model_part_input_info.input_dim],
                **arg)

        elif part_name == E_arch_position.backbone_with_neck.name:  # Backbone_with_Neck
            return backbone_with_neck.backbone_with_neck_factory_[model_name](**arg)

        elif part_name == E_arch_position.backbone.name:  # Backbone
            return backbone.backbone_factory_[model_name](**arg)

        elif part_name == E_arch_position.neck.name:  # Neck
            return neck.neck_factory_[model_name](
                input_dim=self.input_info[E_arch_position.neck.name][E_model_part_input_info.input_dim],
                **arg
            )

        else:
            self.logger.info('Unvalid part name')
            raise NotImplementedError

    def get_part_input_info(self, name, modules_dict, input_size=1024):
        x = Variable(torch.randn((1, 3, input_size, input_size)))
        model = nn.Sequential(modules_dict)
        net = model.eval()
        results = list(net(x))

        self.input_info[name] = dict()
        self.input_info[name][E_model_part_input_info.input_dim] = [result.size()[1] for result in results]
        self.input_info[name][E_model_part_input_info.scale] = [int(input_size / result.size()[2]) for result in results]

        del net, x
        torch.cuda.empty_cache()


class S_info_data:
    def __init__(self, model: BaseModel):
        self.model = model
        self.mean = [0.408, 0.447, 0.470]
        self.std = [0.289, 0.274, 0.278]
        self.nID_dict = {}
        self.input_info = model.input_info
        self.classes_max_num = model.classes_max_num
        self.objects_max_num = model.objects_max_num

    def update_dataset_info(self, dataset):
        self.mean, self.std = dataset.mean, dataset.std
        self.classes_num = dataset.num_classes if dataset.num_classes > self.model.classes_max_num else self.classes_max_num
        self.nID_dict = dataset.nID_dict

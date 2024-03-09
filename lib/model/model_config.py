from lib.utils.yacs import CfgNode as CN
from .networks.part_config_master import part_cfg_master as part_cfg
from .networks.part_config_master import E_part_info
from enum import Enum, unique


@unique
class E_arch_position(Enum):
    head = 0
    backbone_with_neck = 1
    neck = 2
    backbone = 3


@unique
class E_model_general_info(Enum):
    _description = 0
    max_classes_num = 1
    max_objects_num = 2


@unique
class E_model_part_input_info(Enum):
    input_dim = 0
    scale = 1

model_general_info_default_dict = {
    E_model_general_info(0): '',
    E_model_general_info(1): 1,
    E_model_general_info(2): 500,
}

model_arch_dict = {k: part_cfg for k in E_arch_position.__members__}

model_info_save_dict = {k.name: v for k, v in model_general_info_default_dict.items()}
model_info_save_dict.update(model_arch_dict)

model_cfg_master = CN(model_info_save_dict)


def check_model_architecture(cfg: CN):
    bHaveHead = False
    bHaveBackbone_with_Neck = False
    bHaveBackbone = False
    bHaveNeck = False

    for k, v in cfg.items():
        model_name = ''
        if k in E_arch_position.__members__.keys():
            model_name = v[E_part_info(1).name]
        if k == E_arch_position(0).name and model_name != '':
            bHaveHead = True
        elif k == E_arch_position(1).name and model_name != '':
            bHaveBackbone_with_Neck = True
        elif k == E_arch_position(3).name and model_name != '':
            bHaveBackbone = True
        elif k == E_arch_position(2).name and model_name != '':
            bHaveNeck = True

    arch_list = []  # Order of this list is important
    if not bHaveBackbone_with_Neck:
        if not bHaveBackbone:
            raise AttributeError('Model lacks backbone!')
        else:
            arch_list.append(E_arch_position(2).name)
            if not bHaveNeck:
                raise AttributeError('Model only has backbone, lacks neck!')
            else:
                arch_list.append(E_arch_position(3).name)
    else:
        arch_list.append(E_arch_position(1).name)
        if bHaveBackbone or bHaveNeck:
            raise AttributeError('Model had both backbone+neck, backbone or neck! Too many parts!')

    if not bHaveHead:
        raise AttributeError('Model lacks head!')
    else:
        arch_list.append(E_arch_position(0).name)

    return arch_list


def merge_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


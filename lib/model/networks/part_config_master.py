from lib.utils.yacs import CfgNode as CN
from enum import Enum, unique


@unique
class E_part_info(Enum):
    _description = 0
    model_name = 1
    cfg_name = 2


part_cfg_default_dict = {
    E_part_info(0): '',
    E_part_info(1): '',
    E_part_info(2): '',
}

part_cfg_master = CN({k.name: v for k, v in part_cfg_default_dict.items()})

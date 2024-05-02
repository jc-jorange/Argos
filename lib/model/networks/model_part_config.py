from lib.utils.yacs import CfgNode as CN
from enum import Enum, unique


@unique
class E_model_part_info(Enum):
    _description = 0
    model_name = 1
    cfg_name = 2


dict_model_part_default = {
    E_model_part_info._description: '',
    E_model_part_info.model_name: '',
    E_model_part_info.cfg_name: '',
}

model_part_cfg_master = CN({k.name: v for k, v in dict_model_part_default.items()})

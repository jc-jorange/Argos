from lib.utils.yacs import CfgNode as CN
from enum import Enum, unique


@unique
class E_pipeline_group(Enum):
    indis = 0
    globals = 1


@unique
class E_pipeline_station(Enum):
    producer = 0
    consumer = 1
    post = 2


dict_pipeline_position = {k: {} for k in E_pipeline_station.__members__}
dict_pipeline_group = {k: dict_pipeline_position for k in E_pipeline_group.__members__}
pipeline_cfg_master = CN(dict_pipeline_group)

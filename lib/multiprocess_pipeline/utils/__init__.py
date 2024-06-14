from yacs.config import CfgNode as CN

from ..process import E_pipeline_branch
from ..process import factory_process_all
from ..SharedMemory import E_PipelineSharedDataName, E_SharedDataFormat, E_SharedSaveType
from ..SharedMemory import dict_SharedDataInfoFormat


def check_pipeline_cfg(cfg_dir: str) -> None:
    with open(cfg_dir, 'r') as pipeline_cfg:
        process_yaml = CN.load_cfg(pipeline_cfg)

        for pipeline_name, pipeline in process_yaml.items():
            for pipeline_branch_name, pipeline_branch in pipeline.items():
                check_pipeline_branch(pipeline_name, pipeline_branch_name)

                if pipeline_branch:
                    if pipeline_branch_name == E_pipeline_branch.static_shared_value.name:
                        for shared_value_name, shared_value_content in pipeline_branch.items():
                            check_initial_shared_value(pipeline_name, shared_value_name, shared_value_content)
                    else:
                        for pipeline_leaf_name, pipeline_leaf in pipeline_branch.items():
                            check_pipeline_leaf(pipeline_name,
                                                pipeline_branch_name,
                                                pipeline_leaf_name,)


def check_pipeline_branch(pipeline_name: str, branch_name: str) -> None:
    if branch_name in E_pipeline_branch.__members__.keys():
        pass
    else:
        raise ValueError(f'{branch_name} @ pipeline{pipeline_name} not a valid pipeline branch!\n'
                         f'Valid pipeline branches as following:\n'
                         f'{E_pipeline_branch.__members__.keys()}')


def check_pipeline_leaf(pipeline_name: str,
                        branch_name: str,
                        leaf_name: str,) -> None:
    factory_check: dict
    factory_check = factory_process_all[branch_name]
    if leaf_name in factory_check.keys():
        pass
    else:
        raise ValueError(f'{leaf_name} in branch {branch_name} @ {pipeline_name} not a valid process!\n'
                         f'Valid leaf processes in {branch_name} as following:\n'
                         f'{factory_check.keys()}')


def check_initial_shared_value(pipeline_name: str, value_name: str, content: dict) -> None:
    if value_name in E_PipelineSharedDataName.__members__.keys():
        for content_name, content_value in content.items():
            if content_name in E_SharedDataFormat.__members__.keys():
                if content_name == E_SharedDataFormat.data_type.name:
                    try:
                        dict_SharedDataInfoFormat[content_name]
                    except KeyError:
                        raise ValueError(f'{content_name} in {value_name} @ pipeline {pipeline_name} '
                                         f'is not a valid shared data type!\n'
                                         f'Valid shared data type as following:\n'
                                         f'{E_SharedSaveType.__members__.keys()}')

                elif content_name == E_SharedDataFormat.data_shape.name:
                    try:
                        tuple(content_value)
                    except TypeError:
                        raise ValueError(f'{content_name} in {value_name} @ pipeline {pipeline_name} is not iterable')

                elif content_name == E_SharedDataFormat.data_value.name:
                    pass

            else:
                raise ValueError(f'{content_name} in {value_name} @ pipeline {pipeline_name} '
                                 f'is not a valid shared value content!\n'
                                 f'Valid shared value content name as following:\n'
                                 f'{E_SharedDataFormat.__members__.keys()}')

    else:
        raise ValueError(f'{value_name} @ pipeline {pipeline_name} '
                         f'is not a valid shared value!\n'
                         f'Valid shared value name as following:\n'
                         f'{E_PipelineSharedDataName.__members__.keys()}')

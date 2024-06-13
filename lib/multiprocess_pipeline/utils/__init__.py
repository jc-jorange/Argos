from yacs.config import CfgNode as CN

from ..process import E_pipeline_branch


def check_pipeline_cfg(cfg_dir: str) -> bool:
    with open(cfg_dir, 'r') as pipeline_cfg:
        process_yaml = CN.load_cfg(pipeline_cfg)

        for pipeline_name, pipeline_branch in process_yaml.items():
            if pipeline_name in E_pipeline_branch.__members__:
                pass
            else:
                raise ValueError(f'{pipeline_name} not a valid pipeline')

    return True

import multiprocessing as mp
from yacs.config import CfgNode

from .shared_data import Struc_SharedData
import src.multiprocess_pipeline.process as pipe_process


class SharedDataHub:
    def __init__(self,
                 device: str,
                 pipeline_cfg: CfgNode):

        self.dict_shared_data = {}

        for pipeline_name, pipeline_branch in pipeline_cfg.items():
            tmp_shared_data_dict = {}
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                if pipeline_leaf and (pipeline_branch_name in pipe_process.factory_process_all.keys()):
                    for pipeline_leaf_name in pipeline_leaf.keys():
                        for shared_data_name, shared_data_info in \
                                pipe_process.factory_process_all[pipeline_branch_name][pipeline_leaf_name]\
                                .shared_data.items():
                            tmp_shared_data_dict.update({shared_data_name: Struc_SharedData(device, shared_data_info)})
            self.dict_shared_data.update({pipeline_name: tmp_shared_data_dict})

        self.dict_consumer_port = {
            pipeline_name: [] for pipeline_name, pipeline_branch in pipeline_cfg.items()
        }

        self.dict_process_results_dir = {}
        for pipeline_name, pipeline_branch in pipeline_cfg.items():
            tmp_dict_branch = {}
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                tmp_dict_leaf = {}
                if pipeline_leaf:
                    for pipeline_leaf_name in pipeline_leaf.keys():
                        tmp_dict_leaf[pipeline_leaf_name] = ''
                tmp_dict_branch.update({pipeline_branch_name: tmp_dict_leaf})
            self.dict_process_results_dir.update({pipeline_name: tmp_dict_branch})

        self.dict_bLoadingFlag = {
            pipeline_name: mp.Value('b', 1) for pipeline_name in pipeline_cfg.keys()
        }

        self.array_schedule_gpu = mp.Array('i', 2)
        self.array_schedule_gpu[0] = -1

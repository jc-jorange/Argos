import os
import sys
import torch
import multiprocessing as mp
from yacs.config import CfgNode as CN

from src.opts.track import opts_track, argparse
from src.multiprocess_pipeline.shared_structure import SharedDataHub, Struc_SharedData
from src.utils.logger import ALL_LoggerContainer, logging

from src.multiprocess_pipeline.process import ProducerProcess, ConsumerProcess, PostProcess
from src.multiprocess_pipeline.process import E_pipeline_branch
from src.multiprocess_pipeline.process import factory_process_all

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

MAIN_PROCESS_NAME = 'Argus-MainTrackProcess'


def track(opt_data: argparse.Namespace,
          logger: logging.Logger):
    """
    :param opt_data:
    :param logger:
    :return:
    """
    torch.multiprocessing.set_start_method('spawn')
    if logger:
        track_main_logger = logger
    else:
        track_main_logger = ALL_LoggerContainer.add_logger(mp.current_process().name)

    # Start Tracking
    track_main_logger.info('-' * 10 + 'Start Tracking' + '-' * 10)

    # Log opt content in this experiment
    track_main_logger.info("opt:")
    for k, v in vars(opt_data).items():
        track_main_logger.info('  %s: %s' % (str(k), str(v)))

    # Read multiprocess pipeline config file
    with open(opt_data.pipeline_cfg, 'r') as pipeline_cfg:
        process_yaml = CN.load_cfg(pipeline_cfg)

    # Total indi pipeline
    sub_processor_num = len(process_yaml.keys())
    track_main_logger.info(f'Total {sub_processor_num} pipelines')

    # Initialize multiprocess container for Main process
    pipeline_tree = {
        pipeline_name: {
            branch_name: {} for branch_name, leaf in pipeline_branch.items()
        } for pipeline_name, pipeline_branch in process_yaml.items()
    }

    # Initialize data hub
    data_hub = SharedDataHub(opt_data.device, process_yaml)

    # Creating pipeline
    track_main_logger.info(f'Start creating pipelines')
    for pipeline_name, pipeline_branch in process_yaml.items():
        for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
            last_consumer_port = None

            if pipeline_leaf:
                for pipeline_leaf_name, pipeline_kwargs in pipeline_leaf.items():
                    pipeline_kwargs = dict(pipeline_kwargs) if pipeline_kwargs else {}

                    # producer branch pre-process
                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        pass
                    # consumer branch pre-process
                    elif pipeline_branch_name == E_pipeline_branch.consumer.name:
                        if last_consumer_port:
                            pipeline_kwargs.update({'last_process_port': last_consumer_port})
                    # post branch pre-process
                    elif pipeline_branch_name == E_pipeline_branch.post.name:
                        pass
                    # preprocess initial shared data
                    elif pipeline_branch_name == E_pipeline_branch.static_shared_value.name:
                        track_main_logger.info('-' * 5 + f'Creating '
                                                         f'Pipeline: {pipeline_name},'
                                                         f'shared value {pipeline_leaf_name} ' +
                                               '-' * 5)
                        data_hub.dict_shared_data[pipeline_name].update(
                            {pipeline_leaf_name: Struc_SharedData(opt_data.device, pipeline_kwargs)}
                        )
                        continue

                    track_main_logger.info('-' * 5 + 'Setting ' 
                                           f'Pipeline: {pipeline_name}, '
                                           f'Sub-Processor: {pipeline_leaf_name} @ {pipeline_branch_name} '
                                           + '-' * 5)

                    # processor creating form factory
                    processor = factory_process_all[pipeline_branch_name][pipeline_leaf_name](
                        data_hub=data_hub,
                        pipeline_name=pipeline_name,
                        opt=opt_data,
                        **pipeline_kwargs,
                    )
                    processor.daemon = True
                    processor.start()

                    # producer branch post-process
                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        processor: ProducerProcess
                        pass
                    # consumer branch post-process
                    elif pipeline_branch_name == E_pipeline_branch.consumer.name:
                        processor: ConsumerProcess
                        last_consumer_port = processor.output_port
                        data_hub.dict_consumer_port[pipeline_name].append(last_consumer_port)
                    # post branch post-process
                    elif pipeline_branch_name == E_pipeline_branch.post.name:
                        processor: PostProcess
                        pass

                    # add processor to pipeline tree
                    pipeline_tree[pipeline_name][pipeline_branch_name][pipeline_leaf_name] = processor
                    # add processor result save dir to data_hub
                    data_hub.dict_process_results_dir[pipeline_name][pipeline_branch_name][pipeline_leaf_name] \
                        = processor.results_save_dir

    # actually run all processor
    for pipeline_name, pipeline_branch in pipeline_tree.items():
        for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
            for pipeline_leaf_name, process in pipeline_leaf.items():
                track_main_logger.info('-' * 5 + 'Starting '
                                       f'Pipeline: {pipeline_name}, '
                                       f'Sub-Processor: {pipeline_leaf_name} @ {pipeline_branch_name} '
                                       + '-' * 5)
                process.process_run_action()

    # check all processor running
    b_check_sub = True
    while b_check_sub:
        b_check_sub = False

        for pipeline_name, pipeline_branch in pipeline_tree.items():
            b_check_producer = False  # check each pipeline all producer still receiving data
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                for pipeline_leaf_name, process in pipeline_leaf.items():
                    b_check_sub = b_check_sub or process.is_alive()  # if we have an alive processor, keep check next
                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        # if we have a receiving producer, do nothing. If all producer finish receive, flag False
                        b_check_producer = b_check_producer or process.recv_alive()
                        data_hub.dict_bLoadingFlag[pipeline_name].value = b_check_producer
                        if not b_check_producer:
                            process.kill()  # manually kill finished producer

    track_main_logger.info('-' * 10 + 'Main Finished' + '-' * 10)


if __name__ == '__main__':
    opt = opts_track().init()

    mp.current_process().name = MAIN_PROCESS_NAME

    main_logger = ALL_LoggerContainer.add_logger(MAIN_PROCESS_NAME)
    ALL_LoggerContainer.add_stream_handler(MAIN_PROCESS_NAME)
    ALL_LoggerContainer.add_file_handler(MAIN_PROCESS_NAME, 'Main', opt.save_dir)
    ALL_LoggerContainer.set_logger_level(MAIN_PROCESS_NAME, 'debug' if opt.debug else 'info')

    main_logger.info(f'==> torch version: {torch.__version__}')
    main_logger.info(f'==> cudnn version: {torch.backends.cudnn.version()}')
    main_logger.info('==> Cmd:')
    main_logger.info(str(sys.argv))

    torch.cuda.empty_cache()

    track(opt, main_logger)

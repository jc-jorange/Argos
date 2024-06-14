import multiprocessing as mp
import os
import sys
import json

from yacs.config import CfgNode as CN
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.multiprocess_pipeline.SharedMemory import SharedDataHub, Struc_SharedData
from lib.opts import opts
from lib.model import load_model, save_model, BaseModel
from lib.utils.logger import ALL_LoggerContainer
from lib.dataset import TrainingDataset
from lib.trainer import BaseTrainer

from lib.multiprocess_pipeline.process import ProducerProcess, ConsumerProcess, PostProcess
from lib.multiprocess_pipeline.process import mkdir_if_missing
from lib.multiprocess_pipeline.process import E_pipeline_branch
from lib.multiprocess_pipeline.process import factory_process_all

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

MAIN_PROCESS_NAME = 'Argus-MainProcess'
TENSORBOARD_WRITER_NAME = 'Argus-Train-TensorboardWriter'


def train(opt_data):
    ALL_LoggerContainer.add_tensorboard_writer(TENSORBOARD_WRITER_NAME, opt.save_dir)

    torch.manual_seed(opt_data.seed)
    torch.backends.cudnn.benchmark = not opt_data.not_cuda_benchmark

    main_logger.info('-' * 10 + 'Start Training' + '-' * 10)

    main_logger.info("opt:")
    for k, v in vars(opt_data).items():
        main_logger.info(f'  {k}: {v}')

    main_logger.info('-' * 5 + 'Setting up data...')
    with open(opt_data.data_cfg) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']  # dataset training files path
        valset_paths = data_config['test']
        dataset_root = data_config['root']  # dataset root dir
        main_logger.info(f"Dataset root: {dataset_root}")

    # Image data transformations
    transforms = T.Compose([T.ToTensor()])

    # create model
    main_logger.info('-' * 5 + 'Creating model...')
    model = BaseModel(opt_data)

    # Dataset
    main_logger.info('-' * 5 + 'Setting dataset...')
    train_dataset = TrainingDataset(opt=opt_data,
                                    info_data=model.info_data,
                                    root=dataset_root,
                                    paths=trainset_paths,
                                    augment=True,
                                    transforms=transforms)
    val_dataset = TrainingDataset(opt=opt_data,
                                  info_data=model.info_data,
                                  root=dataset_root,
                                  paths=valset_paths,
                                  augment=True,
                                  transforms=transforms)
    model.info_data.update_dataset_info(train_dataset)

    # initial optimizer
    main_logger.info('-' * 5 + 'Creating optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), opt_data.lr)

    main_logger.info('-' * 5 + 'Creating trainer...')
    start_epoch = 0
    if opt_data.load_model != '':
        main_logger.info('-' * 5 + 'Loading model...')
        model, optimizer, start_epoch = load_model(model,
                                                   opt_data.load_model,
                                                   optimizer,
                                                   opt_data.resume,
                                                   opt_data.lr,
                                                   opt_data.lr_step)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt_data.batch_size,
                                               num_workers=opt_data.num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=opt_data.batch_size,
                                             num_workers=opt_data.num_workers,
                                             pin_memory=False,
                                             drop_last=True)

    trainer = BaseTrainer(opt=opt_data, model=model, optimizer=optimizer)
    trainer.set_device(opt_data.gpus, opt_data.chunk_sizes, opt_data.device)

    ALL_LoggerContainer.dump_cfg(mp.current_process().name, model.cfg)

    main_logger.info('-' * 5 + 'Starting training...')
    for epoch in range(start_epoch + 1, start_epoch + opt_data.num_epochs + 1):

        # Train an epoch
        log_dict_train = trainer.train(epoch, train_loader)

        # Logging train
        epoch_result_info = 'train | epoch: {} |'.format(epoch)
        # mian_logger.info('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            ALL_LoggerContainer.scalar_summary_to_tensorboard(TENSORBOARD_WRITER_NAME, f'train_{k}', v, epoch)
            epoch_result_info += f'{k} {v:.8f} | '
            # mian_logger.info('{} {:8f} | '.format(k, v))
        main_logger.info(epoch_result_info)

        # Validation
        if opt_data.val_intervals > 0 and not epoch % opt_data.val_intervals:
            log_dict_train = trainer.val(epoch, val_loader)

            # Logging validation
            epoch_result_info = 'val | epoch: {} |'.format(epoch)
            for k, v in log_dict_train.items():
                ALL_LoggerContainer.scalar_summary_to_tensorboard(TENSORBOARD_WRITER_NAME, f'val_{k}', v, epoch)
                epoch_result_info += f'{k} {v:.8f} | '
            main_logger.info(epoch_result_info)

        # Save model
        if opt_data.save_epochs > 0 and epoch % opt_data.save_epochs == 0:
            save_model(os.path.join(opt_data.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        save_model(os.path.join(opt_data.save_dir, opt_data.arch + '.pth'),
                   epoch, model, optimizer)

        if epoch in opt_data.lr_step:
            save_model(os.path.join(opt_data.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt_data.lr * (0.1 ** (opt_data.lr_step.index(epoch) + 1))
            main_logger.info(f'Drop LR to {lr}')

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    main_logger.info('-' * 5 + 'Finished')


def track(opt_data):
    """
    :param opt_data:
    :return:
    """
    torch.multiprocessing.set_start_method('spawn')

    # Main save directory
    result_dir = opt_data.save_dir
    mkdir_if_missing(result_dir)

    # Start Tracking
    main_logger.info('-' * 10 + 'Start Tracking' + '-' * 10)

    # Log opt content in this experiment
    main_logger.info("opt:")
    for k, v in vars(opt_data).items():
        main_logger.info('  %s: %s' % (str(k), str(v)))

    # Read multiprocess pipeline config file
    with open(opt_data.pipeline_cfg, 'r') as pipeline_cfg:
        process_yaml = CN.load_cfg(pipeline_cfg)

    # Total indi pipeline
    sub_processor_num = len(process_yaml.keys())
    main_logger.info(f'Total {sub_processor_num} pipelines')

    # Initialize multiprocess container for Main process
    pipeline_tree = {
        pipeline_name: {
            branch_name: {}
            for branch_name, leaf in pipeline_branch.items()
        } for pipeline_name, pipeline_branch in process_yaml.items()
    }

    # Initialize data hub
    data_hub = SharedDataHub(opt_data.device, process_yaml)

    for pipeline_name, pipeline_branch in process_yaml.items():
        for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
            last_consumer_port = None

            if pipeline_leaf:
                for pipeline_leaf_name, pipeline_kwargs in pipeline_leaf.items():
                    pipeline_kwargs = dict(pipeline_kwargs) if pipeline_kwargs else {}

                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        pass
                    elif pipeline_branch_name == E_pipeline_branch.consumer.name:
                        if last_consumer_port:
                            pipeline_kwargs.update({'last_process_port': last_consumer_port})
                    elif pipeline_branch_name == E_pipeline_branch.post.name:
                        pass
                    elif pipeline_branch_name == E_pipeline_branch.static_shared_value.name:
                        data_hub.dict_shared_data[pipeline_name].update(
                            {pipeline_leaf_name: Struc_SharedData(opt_data.device, pipeline_kwargs)}
                        )
                        continue

                    main_logger.info('-' * 5 + 'Setting ' 
                                               f'Pipeline: {pipeline_name}, '
                                               f'Sub-Processor: {pipeline_leaf_name} @ {pipeline_branch_name} '
                                     + '-' * 5)

                    processor = factory_process_all[pipeline_branch_name][pipeline_leaf_name](
                        data_hub=data_hub,
                        pipeline_name=pipeline_name,
                        opt=opt_data,
                        **pipeline_kwargs,
                    )

                    processor.start()

                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        processor: ProducerProcess
                        pass
                    elif pipeline_branch_name == E_pipeline_branch.consumer.name:
                        processor: ConsumerProcess
                        last_consumer_port = processor.output_port
                        data_hub.dict_consumer_port[pipeline_name].append(last_consumer_port)
                    elif pipeline_branch_name == E_pipeline_branch.post.name:
                        processor: PostProcess
                        pass

                    pipeline_tree[pipeline_name][pipeline_branch_name][pipeline_leaf_name] = processor
                    data_hub.dict_process_results_dir[pipeline_name][pipeline_branch_name][pipeline_leaf_name] \
                        = processor.results_save_dir

    for pipeline_name, pipeline_branch in pipeline_tree.items():
        for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
            for pipeline_leaf_name, process in pipeline_leaf.items():
                main_logger.info('-' * 5 + 'Starting '
                                           f'Pipeline: {pipeline_name}, '
                                           f'Sub-Processor: {pipeline_leaf_name} @ {pipeline_branch_name} '
                                 + '-' * 5)
                process.process_run_action()

    b_check_sub = True
    while b_check_sub:
        b_check_sub = False
        b_check_producer = False
        for pipeline_name, pipeline_branch in pipeline_tree.items():
            for pipeline_branch_name, pipeline_leaf in pipeline_branch.items():
                for pipeline_leaf_name, process in pipeline_leaf.items():
                    b_check_sub = b_check_sub or process.is_alive()
                    if pipeline_branch_name == E_pipeline_branch.producer.name:
                        b_check_producer = b_check_producer or process.is_alive()
                data_hub.dict_bLoadingFlag[pipeline_name].value = b_check_producer

    main_logger.info('-' * 10 + 'Main Finished' + '-' * 10)


if __name__ == '__main__':
    opt = opts().init()

    mp.current_process().name = MAIN_PROCESS_NAME

    main_logger = ALL_LoggerContainer.add_logger(MAIN_PROCESS_NAME)
    ALL_LoggerContainer.add_stream_handler(MAIN_PROCESS_NAME)
    ALL_LoggerContainer.add_file_handler(MAIN_PROCESS_NAME, 'Main', opt.save_dir)
    ALL_LoggerContainer.set_logger_level(MAIN_PROCESS_NAME, 'debug' if opt.debug else 'info')

    main_logger.info(f'==> torch version: {torch.__version__}')
    main_logger.info(f'==> cudnn version: {torch.backends.cudnn.version()}')
    main_logger.info('==> Cmd:')
    main_logger.info(str(sys.argv))

    if opt.train:
        train(opt)
    else:
        track(opt)

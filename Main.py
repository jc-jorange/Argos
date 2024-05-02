import multiprocessing as mp
import os
import sys
from collections import defaultdict
import json
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.multiprocess_pipeline.SharedMemory import DataHub
from lib.opts import opts
from lib.model import load_model, save_model, BaseModel
from lib.utils.logger import ALL_LoggerContainer
from lib.dataset import TrainingDataset
from lib.trainer import BaseTrainer
from lib.multiprocess_pipeline.workers.tracker.utils.utils import mkdir_if_missing
from lib.multiprocess_pipeline.process_group.individual_process.producer import factory_indi_process_producer
from lib.multiprocess_pipeline.process_group.individual_process.consumer import factory_indi_process_consumer, E_Indi_Process_Consumer
from lib.multiprocess_pipeline.process_group.individual_process.post import factory_indi_process_post, E_Indi_Process_Post
from lib.multiprocess_pipeline.process_group.global_process import factory_global_process_producer
from lib.multiprocess_pipeline.process_group.global_process.consumer import factory_global_process_consumer
from lib.multiprocess_pipeline.process_group.global_process import factory_global_process_post
from lib.multiprocess_pipeline.SharedMemory import E_ProducerOutputName_Indi, E_ProducerOutputName_Global

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

    result_dir = opt_data.save_dir
    mkdir_if_missing(result_dir)

    main_logger.info('-' * 10 + 'Start Tracking' + '-' * 10)

    main_logger.info("opt:")
    for k, v in vars(opt_data).items():
        main_logger.info('  %s: %s' % (str(k), str(v)))

    container_indi_multiprocess = defaultdict(dict)
    container_global_multiprocess = {}
    container_indi_multiprocess_dir = defaultdict(dict)
    container_global_multiprocess_dir = {}
    indi_last_consumer_dict = {}

    sub_processor_num = len(opt_data.input_path)

    data_hub = DataHub(opt_data, sub_processor_num)

    main_logger.info(f'Total {sub_processor_num} cameras are loaded')

    for model_index in range(sub_processor_num):
        for process_type, process_class in factory_indi_process_producer.items():
            main_logger.info('-' * 5 + f'Setting {process_type} Sub-Processor {model_index}' + '-' * 5)
            processor = process_class(data_hub, model_index, opt_data)
            container_indi_multiprocess[model_index][process_type] = processor
            processor.start()

            container_indi_multiprocess_dir[model_index][process_type] = processor.main_save_dir

        last_consumer_port = None
        for process_type, process_class in factory_indi_process_consumer.items():
            main_logger.info('-' * 5 + f'Setting {process_type} Sub-Processor {model_index}' + '-' * 5)
            processor = process_class(
                producer_result_hub=data_hub,
                idx=model_index,
                opt=opt_data,
                last_process_port=last_consumer_port,
            )
            last_consumer_port = processor.output_port
            data_hub.consumer_port[model_index].append(processor.output_port)
            indi_last_consumer_dict[model_index] = last_consumer_port
            container_indi_multiprocess[model_index][process_type] = processor
            processor.start()

            container_indi_multiprocess_dir[model_index][process_type] = processor.main_save_dir

    for process_type, process_class in factory_global_process_producer.items():
        main_logger.info('-' * 5 + f'Setting {process_type} Global-Processor' + '-' * 5)
        processor = process_class(
            data_hub, 0, opt_data
        )
        container_global_multiprocess[process_type] = processor
        processor.start()

        container_global_multiprocess_dir[process_type] = processor.main_save_dir

    last_consumer_port = None
    for process_type, process_class in factory_global_process_consumer.items():
        main_logger.info('-' * 5 + f'Setting {process_type} Global-Processor' + '-' * 5)
        processor = process_class(
            producer_result_hub=data_hub,
            idx=0,
            opt=opt_data,
            last_process_port=last_consumer_port,
        )
        last_consumer_port = processor.output_port
        container_global_multiprocess[process_type] = processor
        processor.start()

        container_global_multiprocess_dir[process_type] = processor.main_save_dir

    for model_index, each_index_process in container_indi_multiprocess.items():
        for process_type, process_class in each_index_process.items():
            process_ = process_class
            main_logger.info('-' * 5 + f'Start {process_class} Sub-Process No.{model_index}' + '-' * 5)
            process_.process_run_action()

    for process_type, process_class in container_global_multiprocess.items():
        process_ = process_class
        main_logger.info('-' * 5 + f'Start {process_class} Global-Process ' + '-' * 5)
        process_.process_run_action()

    b_check_tracker = True
    while b_check_tracker:
        b_check_tracker = False
        for i, p in container_indi_multiprocess.items():
            b_check_tracker = b_check_tracker or p[E_Indi_Process_Consumer.Tracker.name].is_alive()
            container_indi_producer_hub_dict[i].producer_data[E_ProducerOutputName_Indi.bInputLoading].value = b_check_tracker
    global_producer_hub.producer_data[E_ProducerOutputName_Global.bInputLoading].value = False

    for model_index in range(sub_processor_num):
        for process_type, process_class in factory_indi_process_post.items():
            main_logger.info('-' * 5 + f'Setting {process_type} Post-Processor {model_index}' + '-' * 5)
            processor = process_class(
                producer_result_hub=data_hub,
                process_dir=container_indi_multiprocess_dir,
                idx=model_index + 1,
                opt=opt_data
            )
            container_indi_multiprocess[model_index][process_type] = processor
            processor.start()
            processor.process_run_action()

    global_post = None
    for process_type, process_class in factory_global_process_post.items():
        main_logger.info('-' * 5 + f'Setting {process_type} Global-Post-Processor' + '-' * 5)
        processor = process_class(
            producer_result_hub=data_hub,
            indi_process_dir=container_indi_multiprocess_dir,
            global_process_dir=container_global_multiprocess_dir,
            idx=0,
            opt=opt_data)
        # container_multiprocess[sub_processor_num + 1][process_type] = processor
        global_post = processor
        processor.start()
        processor.process_run_action()

    b_check_tracker = True
    while b_check_tracker:
        b_check_tracker = False
        for i, p in container_indi_multiprocess.items():
            b_check_tracker = b_check_tracker or p[E_Indi_Process_Post.IndiPost.name].is_alive()

    while global_post.is_alive():
        pass

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

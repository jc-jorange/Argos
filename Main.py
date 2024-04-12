import multiprocessing as mp
import os
import sys
from collections import defaultdict
import json
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.multiprocess.SharedMemory import ProducerBucket, EQueueType
from lib.opts import opts
from lib.model import load_model, save_model, BaseModel
from lib.utils.logger import ALL_LoggerContainer
from lib.dataset import TrainingDataset
from lib.trainer import BaseTrainer
from lib.tracker.utils.utils import mkdir_if_missing
from lib.multiprocess import process_factory, EMultiprocess
from lib.multiprocess.individual_process.MP_IndiPost import IndividualPostProcess
from lib.multiprocess.global_process.MP_GlobalIdMatch import GlobalIdMatchProcess
from lib.multiprocess.global_process.MP_GlobalPost import GlobalPostProcess
from lib.matchor.MultiCameraMatch.CenterRayIntersect import CenterRayIntersectMatchor

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

    container_multiprocess = defaultdict(dict)
    container_multiprocess_dir = defaultdict(dict)
    container_multiprocess_queue = defaultdict(mp.Queue)
    container_shared_array = {}

    sub_processor_num = len(opt_data.input_path)
    main_logger.info(f'Total {sub_processor_num} cameras are loaded')

    for model_index in range(sub_processor_num):
        shared_container = ProducerBucket(opt_data)
        container_shared_array[model_index] = shared_container
        container_multiprocess_queue[model_index] = shared_container.queue_dict[EQueueType.PredictResultSend]

        for process_type, process_class in process_factory.items():
            main_logger.info('-' * 5 + f'Setting {process_type.name} Sub-Processor {model_index}' + '-' * 5)
            processor = process_class(shared_container, model_index, opt_data)
            container_multiprocess[model_index][process_type] = processor
            processor.start()

            container_multiprocess_dir[model_index][process_type] = processor.main_output_dir

    main_logger.info('-' * 5 + f'Setting Global Id Matching Sub-Processor')
    global_match_shared_container = ProducerBucket(opt_data)
    global_id_matchor = GlobalIdMatchProcess(
        container_multiprocess_queue, CenterRayIntersectMatchor, global_match_shared_container, -1, opt_data
    )
    global_id_matchor.start()

    for process_type, process_class in process_factory.items():
        for model_index in range(sub_processor_num):
            process_ = container_multiprocess[model_index][process_type]
            main_logger.info('-' * 5 + f'Start {process_class.name} Sub-Process No.{model_index}' + '-' * 5)
            process_.process_run_action()

    main_logger.info('-' * 5 + f'Start Global Id Matching Sub-Processor')
    global_id_matchor.process_run_action()

    b_check_tracker = True
    while b_check_tracker:
        b_check_tracker = False
        for i, p in container_multiprocess.items():
            b_check_tracker = b_check_tracker or p[EMultiprocess.Tracker].is_alive()

    global_match_shared_container.b_input_loading.value = False

    for model_index in range(sub_processor_num):
        main_logger.info('-' * 5 + f'Setting Individual PostProcess Sub-Processor {model_index}')
        indi_post = IndividualPostProcess(
            container_multiprocess_dir,
            container_shared_array[model_index], model_index, opt_data
        )
        container_multiprocess[model_index][EMultiprocess.IndiPost] = indi_post
        indi_post.start()
        indi_post.process_run_action()

    for i_process, match_result_dir in global_id_matchor.match_result_dir_dict.items():
        container_multiprocess_dir[i_process].update({EMultiprocess.GlobalMatching: match_result_dir})
    main_logger.info('-' * 5 + f'Setting Global PostProcess Sub-Processor')
    global_post = GlobalPostProcess(
        container_multiprocess_dir,
        global_match_shared_container, -1, opt_data
    )
    global_post.start()
    global_post.process_run_action()

    b_check_tracker = True
    while b_check_tracker:
        b_check_tracker = False
        for i, p in container_multiprocess.items():
            b_check_tracker = b_check_tracker or p[EMultiprocess.IndiPost].is_alive()

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

import multiprocessing
import os
import sys
import torch
import json
import torch.utils.data
from torchvision.transforms import transforms as T

from lib.opts import opts
from lib.model import load_model, save_model, BaseModel
from lib.utils.logger import ALL_LoggerContainer
from lib.dataset import TrainingDataset
from lib.trainer import BaseTrainer
from lib.tracker.utils.utils import mkdir_if_missing
from lib.multiprocess.MP_Tracker import TrackerProcess
from lib.multiprocess.MP_ImageReceiver import ImageReceiverProcess
from lib.multiprocess.MP_PathPredict import PathPredictProcess
import lib.multiprocess.Shared as Sh
from lib.multiprocess.Shared import ESharedDictType
from lib.multiprocess import EMultiprocess
from lib.predictor.spline.hermite_spline import HermiteSpline
import lib.multiprocess.MP_PostIdMatch as MP_Post

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

    ALL_LoggerContainer.dump_cfg(multiprocessing.current_process().name, model.cfg)

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

    container_multiprocess = {
        EMultiprocess.ImageReceiver: {},
        EMultiprocess.Tracker: {},
        EMultiprocess.Predictor: {}
    }

    container_shared_dict = {
        ESharedDictType.Image: {},
        ESharedDictType.Track: {},
        ESharedDictType.Predict: {}
    }

    sub_processor_num = len(opt_data.input_path)

    main_logger.info(f'Total {sub_processor_num} cameras are loaded')
    for model_index in range(sub_processor_num):

        shared_dict_image = Sh.SharedDict()
        container_shared_dict[ESharedDictType.Image][model_index] = shared_dict_image

        shared_dict_track = Sh.SharedDict()
        container_shared_dict[ESharedDictType.Track][model_index] = shared_dict_track

        shared_dict_predict = Sh.SharedDict()
        container_shared_dict[ESharedDictType.Predict][model_index] = shared_dict_predict

        if opt_data.input_mode == 'Address':
            main_logger.info('-' * 5 + f'Setting Image Receiver Sub-Processor {model_index}')
            img_receiver = ImageReceiverProcess(model_index, opt_data, container_shared_dict)
            container_multiprocess[EMultiprocess.ImageReceiver][model_index] = img_receiver

        main_logger.info('-' * 5 + f'Setting Tracker Sub-Processor {model_index}')
        tracker = TrackerProcess(model_index, opt_data, container_shared_dict)
        container_multiprocess[EMultiprocess.Tracker][model_index] = tracker

        main_logger.info('-' * 5 + f'Setting Predictor Sub-Processor {model_index}')
        path_predictor = PathPredictProcess(HermiteSpline, model_index, opt_data, container_shared_dict,)
        container_multiprocess[EMultiprocess.Predictor][model_index] = path_predictor

    for i, p in container_multiprocess[EMultiprocess.ImageReceiver].items():
        main_logger.info('-' * 5 + f'Start Image Receiver Sub-Process No.{i}')
        p.start()

    for i, p in container_multiprocess[EMultiprocess.Tracker].items():
        main_logger.info('-' * 5 + f'Start Tracker Sub-Process No.{i}')
        p.start()

    for i, p in container_multiprocess[EMultiprocess.Predictor].items():
        main_logger.info('-' * 5 + f'Start Predictor Sub-Process No.{i}')
        p.start()

    b_check_tracker = True
    while b_check_tracker:
        b_check_tracker = False
        for i, p in container_multiprocess[EMultiprocess.Tracker].items():
            b_check_tracker = b_check_tracker or p.is_alive()

    for i, each in container_multiprocess[EMultiprocess.Predictor].items():
        each.terminate()

    for i, each in container_multiprocess[EMultiprocess.ImageReceiver].items():
        each.terminate()

    main_logger.info('-' * 10 + 'Main Finished' + '-' * 10)


if __name__ == '__main__':
    opt = opts().init()
    # Set available GPU index

    multiprocessing.current_process().name = MAIN_PROCESS_NAME

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

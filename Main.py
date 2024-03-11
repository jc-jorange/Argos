import multiprocessing
import time
from multiprocessing import Pipe, Lock, Value, shared_memory
import os
import torch
import json

import torch.utils.data
from torchvision.transforms import transforms as T

from lib.opts import opts
from lib.model import load_model, save_model, BaseModel
from lib.utils.logger import logger, add_main_logger
from lib.dataset import TrainingDataset
from lib.trainer import BaseTrainer
from lib.tracker.utils.utils import mkdir_if_missing
from lib.utils.utils import select_device
from lib.multiprocess.MP_Tracker import Tracker_Process
from lib.multiprocess.MP_ImageReceiver import ImageReceiver
from lib.multiprocess.MP_PathPredict import PathPredictProcess
import lib.multiprocess.Shared as Sh
import lib.multiprocess.MP_MultiCameraPredict as MP_Post

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
NAME_shm_img = Sh.NAME_shm_img

def train(opt):
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark

    main_logger.info('-' * 10 + 'Start Training' + '-' * 10)

    main_logger.info("opt:")
    for k, v in vars(opt).items():
        main_logger.info('  {}: {}'.format(k, v))

    main_logger.info('-' * 5 + 'Setting up data...')
    with open(opt.data_cfg) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']  # dataset training files path
        valset_paths = data_config['test']
        dataset_root = data_config['root']  # dataset root dir
        main_logger.info("Dataset root: {}" .format(dataset_root))

    # Image data transformations
    transforms = T.Compose([T.ToTensor()])

    # create model
    main_logger.info('-' * 5 + 'Creating model...')
    model = BaseModel(opt)

    # Dataset
    main_logger.info('-' * 5 + 'Setting dataset...')
    train_dataset = TrainingDataset(opt=opt,
                              info_data=model.info_data,
                              root=dataset_root,
                              paths=trainset_paths,
                              augment=True,
                              transforms=transforms)
    val_dataset = TrainingDataset(opt=opt,
                              info_data=model.info_data,
                              root=dataset_root,
                              paths=valset_paths,
                              augment=True,
                              transforms=transforms)
    model.info_data.update_dataset_info(train_dataset)

    # initial optimizer
    main_logger.info('-' * 5 + 'Creating optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    main_logger.info('-' * 5 + 'Creating trainer...')
    start_epoch = 0
    if opt.load_model != '':
        main_logger.info('-' * 5 + 'Loading model...')
        model, optimizer, start_epoch = load_model(model,
                                                   opt.load_model,
                                                   optimizer,
                                                   opt.resume,
                                                   opt.lr,
                                                   opt.lr_step)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.num_workers,
                                               pin_memory=False,
                                               drop_last=True)

    trainer = BaseTrainer(opt=opt, model=model, optimizer=optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    logger.dump_cfg(os.getpid(), model.cfg)

    main_logger.info('-' * 5 + 'Starting training...')
    for epoch in range(start_epoch + 1, start_epoch + opt.num_epochs + 1):

        # Train an epoch
        log_dict_train = trainer.train(epoch, train_loader)

        # Logging train
        epoch_result_info = 'train | epoch: {} |'.format(epoch)
        # mian_logger.info('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary_to_tensorboard('train_{}'.format(k), v, epoch)
            epoch_result_info += '{} {:8f} | '.format(k, v)
            # mian_logger.info('{} {:8f} | '.format(k, v))
        main_logger.info(epoch_result_info)

        # Validation
        if opt.val_intervals > 0 and not epoch % opt.val_intervals:
            log_dict_train = trainer.val(epoch, val_loader)

            # Logging validation
            epoch_result_info = 'val | epoch: {} |'.format(epoch)
            for k, v in log_dict_train.items():
                logger.scalar_summary_to_tensorboard('val_{}'.format(k), v, epoch)
                epoch_result_info += '{} {:8f} | '.format(k, v)
            main_logger.info(epoch_result_info)

        # Save model
        if opt.save_epochs > 0 and epoch % opt.save_epochs == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
        save_model(os.path.join(opt.save_dir, opt.arch + '.pth'),
                   epoch, model, optimizer)

        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            main_logger.info('Drop LR to {}'.format(lr))

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    main_logger.info('-' * 5 + 'Finished')

def track(opt):
    """
    :param opt:
    :return:
    """
    torch.multiprocessing.set_start_method('spawn')

    result_dir = opt.save_dir
    mkdir_if_missing(result_dir)

    main_logger.info('-' * 10 + 'Start Tracking' + '-' * 10)

    main_logger.info("opt:")
    for k, v in vars(opt).items():
        main_logger.info('  %s: %s' % (str(k), str(v)))

    sub_processor_num = len(opt.input_path)

    mp_image_receiver_dict = {}
    mp_trackers_dict = {}
    mp_path_predictor_dict = {}

    sd_image_dict = {}
    sd_predict_dict = {}

    main_logger.info('Total {} cameras are loaded'.format(sub_processor_num))

    for model_index in range(sub_processor_num):
        if opt.input_mode == 'Address':
            main_logger.info('-' * 5 + 'Setting Image Receiver Sub-Processor {}'.format(model_index))

            shared_dict_image = Sh.SharedDict()
            sd_image_dict[model_index] = shared_dict_image

            img_receiver = ImageReceiver(model_index, opt, shared_dict_image)
            mp_image_receiver_dict[model_index] = img_receiver

        main_logger.info('-' * 5 + 'Setting Tracker Sub-Processor {}'.format(model_index))

        if sd_image_dict:
            shared_dict_image = sd_image_dict[model_index]
        else:
            shared_dict_image = Sh.SharedDict()
            sd_image_dict[model_index] = shared_dict_image

        tracker = Tracker_Process(model_index, opt, shared_dict_image)
        mp_trackers_dict[model_index] = tracker

        main_logger.info('-' * 5 + 'Setting Predictor Sub-Processor {}'.format(model_index))

        shared_dict_predict = Sh.SharedDict()
        sd_predict_dict[model_index] = shared_dict_predict

        path_predictor = PathPredictProcess(model_index, opt,shared_dict_predict)
        mp_path_predictor_dict[model_index] = path_predictor

    for i, p in mp_image_receiver_dict.items():
        main_logger.info('-' * 5 + 'Start Image Receiver Sub-Process No.{}'.format(i))
        p.start()

    for i, p in mp_trackers_dict.items():
        main_logger.info('-' * 5 + 'Start Tracker Sub-Process No.{}'.format(i))
        p.start()

    for i, p in mp_path_predictor_dict.items():
        main_logger.info('-' * 5 + 'Start Predictor Sub-Process No.{}'.format(i))
        p.start()

    bCheckTracker = True
    while bCheckTracker:
        bCheckTracker = False
        for i,p in mp_trackers_dict.items():
            bCheckTracker = bCheckTracker or p.is_alive()

    main_logger.info('-' * 10 + 'Main Finished' + '-' * 10)


if __name__ == '__main__':
    opt = opts().init()
    # Set available GPU index
    opt.device = select_device(opt)

    multiprocessing.current_process().name = 'Argus-MainProcess'

    main_logger = add_main_logger(os.getpid(), opt)

    if opt.train:
        train(opt)
    else:
        track(opt)

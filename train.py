import os
import sys
import json
import multiprocessing as mp
import torch.utils.data
import torch.backends.cudnn
from torchvision.transforms import transforms as trs

from src.opts.train import opts_train, argparse
from src.utils.logger import ALL_LoggerContainer, logging

from src.model import load_model, save_model, BaseModel
from src.dataset import TrainingDataset
from src.trainer import BaseTrainer

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

MAIN_PROCESS_NAME = 'Argus-MainTrainProcess'
TENSORBOARD_WRITER_NAME = 'Argus-Train-TensorboardWriter'


def train(opt_data: argparse.Namespace,
          logger: logging.Logger):
    if logger:
        train_main_logger = logger
    else:
        train_main_logger = ALL_LoggerContainer.add_logger(mp.current_process().name)
    ALL_LoggerContainer.add_tensorboard_writer(TENSORBOARD_WRITER_NAME, opt.save_dir)

    # manual set torch seed
    torch.manual_seed(opt_data.seed)
    # set torch to search better convolution algorithms
    torch.backends.cudnn.benchmark = not opt_data.not_cuda_benchmark

    train_main_logger.info('-' * 10 + 'Start Training' + '-' * 10)

    # Log opt content in this experiment
    train_main_logger.info("opt:")
    for k, v in vars(opt_data).items():
        train_main_logger.info(f'  {k}: {v}')

    # Load data path from json file
    train_main_logger.info('-' * 5 + 'Loading data path...')
    with open(opt_data.data_cfg) as f:
        data_config = json.load(f)
        trainset_paths = data_config['train']  # dataset training files path
        valset_paths = data_config['test']
        dataset_root = data_config['root']  # dataset root dir
        train_main_logger.info(f"Dataset root: {dataset_root}")

    # Image data transformations merge
    transforms = trs.Compose([trs.ToTensor()])

    # Create model
    train_main_logger.info('-' * 5 + 'Creating model...')
    model = BaseModel(opt_data, opt_data.arch)

    # Create Dataset object
    train_main_logger.info('-' * 5 + 'Setting dataset...')
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
    # Update model info based on dataset
    model.info_data.update_dataset_info(train_dataset)

    # initial optimizer
    train_main_logger.info('-' * 5 + 'Creating optimizer...')
    optimizer = torch.optim.Adam(model.parameters(), opt_data.lr)  # Using Adam optimizer

    start_epoch = 0
    if opt_data.load_model != '':
        train_main_logger.info('-' * 5 + 'Loading model...')
        # Model and optimizer setup
        model, optimizer, start_epoch = load_model(model,
                                                   opt_data.load_model,
                                                   optimizer,
                                                   opt_data.resume,
                                                   opt_data.lr,
                                                   opt_data.lr_step)

    # pass dataset to torch dataloader
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

    train_main_logger.info('-' * 5 + 'Creating trainer...')
    trainer = BaseTrainer(opt=opt_data, model=model, optimizer=optimizer)
    # set trainer device
    trainer.set_device(opt_data.gpus, opt_data.chunk_sizes, opt_data.device)

    train_main_logger.info('Dump model config file')
    ALL_LoggerContainer.dump_cfg(mp.current_process().name, model.cfg)

    train_main_logger.info('-' * 5 + 'Starting training...')
    for epoch in range(start_epoch + 1, start_epoch + opt_data.num_epochs + 1):
        # Train an epoch
        log_dict_train = trainer.train(epoch, train_loader)

        # Logging train
        epoch_result_info = f'train | epoch: {epoch} |'
        for k, v in log_dict_train.items():
            ALL_LoggerContainer.scalar_summary_to_tensorboard(TENSORBOARD_WRITER_NAME, f'train_{k}', v, epoch)
            epoch_result_info += f'{k} {v:.8f} | '
        train_main_logger.info(epoch_result_info)

        # Validation
        if opt_data.val_intervals > 0 and not epoch % opt_data.val_intervals:
            log_dict_train = trainer.val(epoch, val_loader)

            # Logging validation
            epoch_result_info = f'val | epoch: {epoch} |'
            for k, v in log_dict_train.items():
                ALL_LoggerContainer.scalar_summary_to_tensorboard(TENSORBOARD_WRITER_NAME, f'val_{k}', v, epoch)
                epoch_result_info += f'{k} {v:.8f} | '
            train_main_logger.info(epoch_result_info)

        # Save model
        if opt_data.save_epochs > 0 and epoch % opt_data.save_epochs == 0:
            save_model(os.path.join(opt_data.save_dir, f'model_{epoch}.pth'), epoch, model, optimizer)
        save_model(os.path.join(opt_data.save_dir, opt_data.arch + '.pth'), epoch, model, optimizer)  # Save every epoch

        # change learning rate
        if epoch in opt_data.lr_step:
            save_model(os.path.join(opt_data.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)

            lr = opt_data.lr * (0.1 ** (opt_data.lr_step.index(epoch) + 1))
            train_main_logger.info(f'Drop LR to {lr}')

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    train_main_logger.info('-' * 5 + 'Finished')


if __name__ == '__main__':
    opt = opts_train().init()

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

    train(opt, main_logger)

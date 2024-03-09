from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
from enum import Enum, unique

import logging

formatter = logging.Formatter(
    # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
    # fmt='%(asctime)s %(processName)s[%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fmt='%(asctime)s %(processName)s[%(levelname)s]: %(message)s')


class Logger_Container:
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }

    @unique
    class logger_name(Enum):
        Main = 0
        Tracker = 1


    def __init__(self):
        self.logger_dict = {}
        self.stream_handler_dict = {}
        self.file_handler_dict = {}
        self.log_dir_dict = {}

    def add_logger(self, log_id):
        local_logger = logging.getLogger(str(log_id))
        self.logger_dict[log_id] = local_logger
        return local_logger

    def add_stream_handler(self, log_id):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger_dict[log_id].addHandler(stream_handler)
        self.stream_handler_dict[log_id] = stream_handler

    def add_file_handler(self, log_id, log_name: str, log_dir: str):
        self.log_dir_dict[log_id] = log_dir
        log_file = log_dir + '/' + log_name + '_log.txt'
        file_handler = logging.FileHandler(filename=log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger_dict[log_id].addHandler(file_handler)
        self.file_handler_dict[log_id] = file_handler

    def set_logger_level(self, log_id, level_name):
        try:
            level = self.level_relations[level_name]
        except KeyError:
            print('None valid name')
            print('valid name as following:')
            print(list(self.level_relations.keys()))
            raise
        else:
            self.logger_dict[log_id].setLevel(level)
            self.logger_dict[log_id].warning('set process-id-{} log level to {}'
                                               .format(self.logger_dict[log_id].name, level_name))

    def scalar_summary_to_tensorboard(self, tag, value, step):
        """Log a scalar variable."""
        if self.USE_TENSORBOARD:
            self.tensorboard_writer.add_scalar(tag, value, step)

    def dump_cfg(self, log_name, cfg_node):
        with open(os.path.join(self.log_dir_dict[log_name], "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)


logger = Logger_Container()


def add_main_logger(name, opt):
    # """Create a summary writer logging to log_dir."""
    # if not os.path.exists(opt.save_dir):
    #     os.makedirs(opt.save_dir)
    # if not os.path.exists(opt.debug_dir):
    #     os.makedirs(opt.debug_dir)

    # time_str = time.strftime('%Y-%m-%d-%H-%M')
    # log_dir = opt.save_dir + '/logs_{}'.format(time_str)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    log_dir = opt.save_dir
    main_logger = logger.add_logger(name)
    logger.add_stream_handler(name)
    logger.add_file_handler(name, 'Main', log_dir)
    logger.set_logger_level(name, 'debug' if opt.debug else 'info')

    main_logger.info('==> torch version: {}'.format(torch.__version__))
    main_logger.info('==> cudnn version: {}'.format(torch.backends.cudnn.version()))
    main_logger.info('==> Cmd:')
    main_logger.info(str(sys.argv))

    if opt.train:
        logger.USE_TENSORBOARD = True
        try:
            import tensorboardX
            main_logger.info('Using tensorboardX')
        except:
            logger.USE_TENSORBOARD = False

        if logger.USE_TENSORBOARD:
            logger.tensorboard_writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
    return main_logger

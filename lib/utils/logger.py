from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
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

    def __init__(self):
        self.logger_dict = {}
        self.stream_handler_dict = {}
        self.file_handler_dict = {}
        self.log_dir_dict = {}
        self.tensorboard_writer_dict = {}

    def add_logger(self, name: str) -> logging.Logger:
        local_logger = logging.getLogger(name + '_logger')
        self.logger_dict[os.getpid()] = local_logger
        return local_logger

    def add_tensorboard_writer(self, name: str, log_dir: str) -> None:
        import tensorboardX
        self.tensorboard_writer_dict[name] = tensorboardX.SummaryWriter(log_dir=log_dir)

    def add_stream_handler(self, logger_name: str) -> None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger_dict[os.getpid()].addHandler(stream_handler)
        self.stream_handler_dict[logger_name] = stream_handler

    def add_file_handler(self, logger_name: str, log_filename: str, log_dir: str) -> None:
        self.log_dir_dict[logger_name] = log_dir
        log_file = log_dir + '/' + log_filename + '_log.txt'
        file_handler = logging.FileHandler(filename=log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger_dict[os.getpid()].addHandler(file_handler)
        self.file_handler_dict[logger_name] = file_handler

    def set_logger_level(self, logger_name: str, level_name: str) -> None:
        try:
            level = self.level_relations[level_name]
        except KeyError:
            print('None valid name')
            print('valid name as following:')
            print(list(self.level_relations.keys()))
            raise
        else:
            self.logger_dict[os.getpid()].setLevel(level)
            self.logger_dict[os.getpid()].warning('set process-id-{} log level to {}'
                                               .format(self.logger_dict[os.getpid()].name, level_name))

    def scalar_summary_to_tensorboard(self, name: str, tag: str, value, step: int) -> None:
        """Log a scalar variable."""
        try:
            tensorboard_writer = self.tensorboard_writer_dict[name]
            tensorboard_writer.add_scalar(tag, value, step)
        except KeyError:
            print('No valid name tensorboard writer')

    def dump_cfg(self, log_name, cfg_node) -> None:
        with open(os.path.join(self.log_dir_dict[log_name], "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)


ALL_LoggerContainer = Logger_Container()

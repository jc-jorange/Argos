from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import logging
from yacs.config import CfgNode as CN
import tkinter

formatter = logging.Formatter(
    # fmt='%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s')
    # fmt='%(asctime)s %(processName)s[%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fmt='%(asctime)s %(processName)s[%(levelname)s]: %(message)s')


class CustomLogger(logging.Logger):
    def __init__(self, name, window: tuple, level=0):
        super(CustomLogger, self).__init__(name, level)
        self.window = window

    def _log(self, level, msg: str, args, exc_info=None, extra=None, stack_info=False, stacklevel=1) -> None:
        super(CustomLogger, self)._log(level, msg, args, exc_info, extra, stack_info, stacklevel)
        text_box = self.window[1]
        text_box.insert(tkinter.END, msg + "\n")
        text_box.see(tkinter.END)
        text_box.update()


class Logger_Container:
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self):
        self.logger_dict = {}
        self.stream_handler_dict = {}
        self.file_handler_dict = {}
        self.log_dir_dict = {}
        self.tensorboard_writer_dict = {}

    def add_logger(self, logger_name: str) -> CustomLogger:
        local_logger = self.get_logger(logger_name)
        if local_logger:
            local_logger.warning(f'Logger {logger_name} already exist in process:{os.getpid()}')
            return local_logger
        else:
            logger_window = tkinter.Tk()
            logger_window.title(logger_name)
            logger_window.protocol("WM_DELETE_WINDOW", lambda: None)
            text = tkinter.Text(logger_window, width=200, height=30)
            text.pack()
            # local_logger = logging.getLogger(logger_name)
            local_logger = CustomLogger(logger_name, (logger_window, text))
            self.logger_dict[logger_name] = local_logger
            return local_logger

    def get_logger(self, logger_name: str) -> CustomLogger | None:
        local = None
        try:
            local = self.logger_dict[logger_name]
        except KeyError:
            return local
        return local

    def add_tensorboard_writer(self, name: str, log_dir: str) -> None:
        import tensorboardX
        self.tensorboard_writer_dict[name] = tensorboardX.SummaryWriter(log_dir=log_dir)

    def add_stream_handler(self, logger_name: str) -> None:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.get_logger(logger_name).addHandler(stream_handler)
        self.stream_handler_dict[logger_name] = stream_handler

    def add_file_handler(self, logger_name: str, log_filename: str, log_dir: str) -> None:
        self.log_dir_dict[logger_name] = log_dir
        log_file = os.path.join(log_dir, log_filename + '_log.txt')
        file_handler = logging.FileHandler(filename=log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.get_logger(logger_name).addHandler(file_handler)
        self.file_handler_dict[logger_name] = file_handler

    def set_logger_level(self, logger_name: str, level_name: str) -> None:
        try:
            level = self.level_relations[level_name]
        except KeyError:
            raise ValueError(f'Not a valid level name\n'
                             f'valid level name as following:\n'
                             f'{self.level_relations}')
        else:
            local_logger = self.get_logger(logger_name)
            local_logger.setLevel(level)
            local_logger.warning(f'set process-id-{logger_name} log level to {level_name}')

    def scalar_summary_to_tensorboard(self, name: str, tag: str, value, step: int) -> None:
        """Log a scalar variable."""
        try:
            tensorboard_writer = self.tensorboard_writer_dict[name]
            tensorboard_writer.add_scalar(tag, value, step)
        except KeyError:
            print('No valid name tensorboard writer')

    def dump_cfg(self, log_name: str, cfg_node: CN) -> None:
        with open(os.path.join(self.log_dir_dict[log_name], "train_cfg.yml"), "w") as f:
            cfg_node.dump(stream=f)


ALL_LoggerContainer = Logger_Container()

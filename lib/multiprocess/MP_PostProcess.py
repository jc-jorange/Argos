import os
import time
import datetime
from multiprocessing import Process
import torch

from lib.utils.logger import logger
from socket import *
import struct


class PostProcess(Process):
    def __init__(self,
                 sub_process_dict: dict,
                 pipe_dict: dict,
                 ):
        super().__init__()
        self.sub_process_dict = sub_process_dict
        self.pipe_dict = pipe_dict

        self.result_dict = {}

    def process_content(self):
        ...

    def process(self):
        self.process_content()

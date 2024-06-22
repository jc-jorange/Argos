from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

from src.utils.utils import select_device


class opts(object):
    def __init__(self):
        self._opt = argparse.Namespace()

        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--exp_id',
                                 default='custom')

        self.parser.add_argument('--debug',
                                 action='store_true',
                                 help='enable log debug mode')

        self.parser.add_argument('--output_root',
                                 type=str,
                                 default='./results',
                                 help='expected output root path')

        # system
        self.parser.add_argument('--gpus',
                                 nargs='+',
                                 default=[0],  # 0, 5, 6
                                 help='-1 for CPU, use comma for multiple gpus')

        self.parser.add_argument('--seed',
                                 type=int,
                                 default=317,
                                 help='random seed')  # from CornerNet

        # model: backbone and so on...
        self.parser.add_argument('--arch_cfg_path',
                                 default='src/model/cfg',
                                 help='model total-all config yaml path')
        self.parser.add_argument('--model_part_path',
                                 default='./src/model/networks',
                                 help='model arch parts path')

    def parse(self, args='') -> argparse.Namespace:
        if args == '':
            self._opt = self.parser.parse_args()
        else:
            self._opt = self.parser.parse_args(args)

        self._opt.num_stacks = 1

        if not os.path.exists(self._opt.output_root):
            os.makedirs(self._opt.output_root)

        self._opt.device = select_device(self._opt.gpus)

        return self._opt

    def init(self, args='') -> argparse.Namespace:
        return self.parse(args)

    def _mk_save_dir(self, save_name: str) -> str:
        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

        save_dir = os.path.join(self._opt.output_root,
                                save_name,
                                self._opt.exp_id,
                                time_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self._opt.save_dir = save_dir
        return save_dir

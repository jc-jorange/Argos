import argparse

from ._base_opt import opts

from src.trainer.utils import check as train_check


class opts_train(opts):
    def __init__(self):
        super(opts, self).__init__()
        self.parser: argparse.ArgumentParser

        # basic experiment setting
        self.parser.add_argument('--resume',
                                 action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=8,  # 8, 6, 4
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark',
                                 action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--gen_multi_scale',
                                 action='store_true',
                                 help='Whether to generate multi-scales')

        # model: backbone and so on...
        self.parser.add_argument('--arch',
                                 default='NanoDet_mot',
                                 help='model architecture. As model config file name')

        # train
        self.parser.add_argument('--data_cfg',
                                 type=str,
                                 default='./src/dataset/cfg/custom.json',
                                 help='load data from cfg')
        self.parser.add_argument('--lr',
                                 type=float,
                                 default=1e-4,  # 1e-4, 7e-5, 5e-5, 3e-5
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step',
                                 nargs='+',
                                 default=[20, 30],  # 20,27
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs',
                                 type=int,
                                 default=30,  # 60, 30, 3, 1
                                 help='total training epochs.')
        self.parser.add_argument('--save_epochs',
                                 type=int,
                                 default=10,
                                 help='number of epochs to save pth.')
        self.parser.add_argument('--batch-size',
                                 type=int,
                                 default=16,  # 18, 16, 14, 12, 10, 8, 4
                                 help='batch size')
        self.parser.add_argument('--master_batch_size',
                                 type=int,
                                 default=-1,
                                 help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters',
                                 type=int,
                                 default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals',
                                 type=int,
                                 default=0,
                                 help='number of epochs to run validation.')

    def parse(self, args=''):
        super(opts_train, self).parse(args=args)

        if self._opt.master_batch_size == -1:
            self._opt.master_batch_size = self._opt.batch_size // len(self._opt.gpus)
        rest_batch_size = (self._opt.batch_size - self._opt.master_batch_size)
        self._opt.chunk_sizes = [self._opt.master_batch_size]
        for i in range(len(self._opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(self._opt.gpus) - 1)
            if i < rest_batch_size % (len(self._opt.gpus) - 1):
                slave_chunk_size += 1
            self._opt.chunk_sizes.append(slave_chunk_size)

        train_check.check_batch_size(self._opt.device, self._opt.batch_size)

        self._mk_save_dir('train_result')

        return self._opt

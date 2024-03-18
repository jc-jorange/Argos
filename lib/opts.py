from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment setting
        self.parser.add_argument('--train',
                                 action='store_true',
                                 help='train or track, default is track')
        self.parser.add_argument('--debug',
                                 action='store_true',
                                 help='enable log debug mode')
        self.parser.add_argument('--exp_id',
                                 default='custom')

        self.parser.add_argument('--load_model',
                                 default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume',
                                 action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        self.parser.add_argument('--input_mode',
                                 type=str,
                                 default='Address',
                                 help='input data type("Video" or "Image" or "Address")')
        self.parser.add_argument('--input_path',
                                 nargs='+',
                                 type=str,
                                 default=['UDP:127.0.0.1:6000'],
                                 help='path or IP address for input')

        self.parser.add_argument('--output_format',
                                 type=str,
                                 default='video',
                                 help='video or text')
        self.parser.add_argument('--output_root',
                                 type=str,
                                 default='./results',
                                 help='expected output root path')

        self.parser.add_argument('--data_cfg',
                                 type=str,
                                 default='./lib/dataset/cfg/custom.json',
                                 help='load data from cfg')

        self.parser.add_argument('--show_image',
                                 action='store_true',
                                 help='whether show result image during tracking')

        # system
        self.parser.add_argument('--gpus',
                                 nargs='+',
                                 default=[0],  # 0, 5, 6
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers',
                                 type=int,
                                 default=8,  # 8, 6, 4
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark',
                                 action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=317,
                                 help='random seed')  # from CornerNet
        self.parser.add_argument('--gen_multi_scale',
                                 action='store_true',
                                 help='Whether to generate multi-scales')

        # model: backbone and so on...
        self.parser.add_argument('--arch',
                                 default='NanoDet_mot',
                                 help='model architecture. As model config file name')
        self.parser.add_argument('--arch_cfg_path',
                                 default='lib/model/cfg',
                                 help='model total-all config yaml path')
        self.parser.add_argument('--part_path',
                                 default='./lib/model/networks',
                                 help='model arch parts path')

        # train
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

        # tracking
        self.parser.add_argument('--conf_thres',
                                 type=float,
                                 default=0.4,  # 0.6, 0.4
                                 help='confidence thresh for tracking')
        self.parser.add_argument('--det_thres',
                                 type=float,
                                 default=0.3,
                                 help='confidence thresh for detection')
        self.parser.add_argument('--nms_thres',
                                 type=float,
                                 default=0.4,
                                 help='iou thresh for nms')
        self.parser.add_argument('--track_buffer',
                                 type=int,
                                 default=30,  # 30
                                 help='tracking buffer')
        self.parser.add_argument('--min-box-area',
                                 type=float,
                                 default=50,
                                 help='filter out tiny boxes')

        # post
        self.parser.add_argument('--post_mode',
                                 type=str,
                                 default='None',
                                 help="'None' for no post-process, 'Outer_TCP' for connect with outer app by TCP/IP, "
                                      "'Outer_UDP' for connect with outer app by UDP ,"
                                      "'Eval' for evaluate results with gt")
        self.parser.add_argument('--post_path',
                                 type=str,
                                 default='127.0.0.1:4000',
                                 help='outer app address')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.num_stacks = 1

        l = []
        for input_item in opt.input_path:
            if ',' in input_item:
                input_item = input_item.split(',')
                input_item = list(filter(None, input_item))
            l.append(input_item)
        opt.input_path = l

        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')

        time_str = time.strftime('%Y-%m-%d-%H-%M-%S')

        opt.save_dir = os.path.join(opt.output_root,
                                    'train_result' if opt.train else 'run_result',
                                    opt.exp_id,
                                    opt.arch,
                                    time_str)
        if not os.path.exists(opt.save_dir):
            os.makedirs(opt.save_dir)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')

        opt.device = self.select_device(opt)

        return opt

    def init(self, args=''):
        opt = self.parse(args)
        return opt

    @staticmethod
    def select_device(opt, apex=False, batch_size=None):
        # device = '[-1]' or '[0]' or '[0,1,2,3]'
        gpus_str = ','.join(str(x) for x in opt.gpus)
        using_cuda = False  # default not use cuda
        if '-1' not in opt.gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
            assert torch.cuda.is_available(), 'CUDA unavailable, invalid device {} requested'.format(gpus_str)  # check cuda
            c = 1024 ** 2  # bytes to MB
            ng = torch.cuda.device_count()
            if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
                assert batch_size % ng == 0, 'batch-size {} not multiple of GPU count {}'.format(batch_size, ng)
            x = [torch.cuda.get_device_properties(i) for i in range(ng)]
            using_cuda = True
            s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
            for i in range(0, ng):
                if i == 1:
                    s = ' ' * len(s)
                print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                      (s, i, x[i].name, x[i].total_memory / c))
        else:
            print('Using CPU')

        print('\n')  # skip a line
        return torch.device('cuda' if using_cuda else 'cpu')


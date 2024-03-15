from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import torch
from progress.bar import Bar
import motmetrics as mm
import torchsummary

from lib.model import BaseModel
from lib.model.data_parallel import DataParallel

from lib.utils.utils import AverageMeter
from lib.utils.logger import ALL_LoggerContainer

from lib.tracker.utils.evaluation import Evaluator


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model: BaseModel, loss):
        super().__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        out = self.model(batch['input'])

        loss, loss_stats = self.loss(out, batch)

        return out[-1], loss, loss_stats


class BaseTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.logger = ALL_LoggerContainer.logger_dict[os.getpid()]
        self.opt = opt
        self.optimizer = optimizer
        self.loss = model.Main.head.loss_class(opt=opt, cfg=model.Main.head.loss_cfg, model_info=model.info_data)
        self.loss_stats = self.loss.cfg['loss_stats']
        # self.loss_stats, self.loss = self._get_losses(opt)
        self.model_to_train = ModelWithLoss(model, self.loss)

        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device):
        # dev_ids = [i for i in range(len(gpus))]
        dev_ids = [int(x) for x in gpus]
        if len(gpus) > 1:
            self.model_to_train = DataParallel(self.model_to_train,
                                               device_ids=dev_ids,  # device_ids=gpus,
                                               chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_to_train = self.model_to_train.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    # Train an epoch
    def run_epoch(self, phase, epoch, data_loader):
        """
        :param phase:
        :param epoch:
        :param data_loader:
        :return:
        """
        model_with_loss = self.model_to_train

        if phase == 'train':
            model_with_loss.train()  # train phase
        else:
            # if len(self.opt.gpus) > 1:
            #     model_with_loss = self.model_to_train.module
            model_with_loss.eval()  # test phase
            torch.cuda.empty_cache()

        opt = self.opt
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader) if opt.num_iters < 0 else opt.num_iters
        ret = {}
        bar = Bar(str(opt.exp_id), max=num_iters)
        end = time.time()

        # train each batch
        # print('Total {} batches in en epoch.'.format(len(data_loader) + 1))
        for batch_i, batch in enumerate(data_loader):
            if batch_i >= num_iters:
                break

            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, batch_i, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            # Forward
            output, loss, loss_stats = model_with_loss(batch)

            # Backwards
            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            # multi-scale img_size display
            scale_idx = data_loader.dataset.batch_i_to_scale_i[batch_i]
            if data_loader.dataset.input_multi_scales is None:
                img_size = [data_loader.dataset.width, data_loader.dataset.height]
            else:
                img_size = data_loader.dataset.input_multi_scales[scale_idx]
            Bar.suffix = Bar.suffix + '|Img_size(wh) {:d}×{:d}'.format(img_size[0], img_size[1])

            Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
                                      '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)

            bar.next()

            del output, loss, loss_stats, batch

            ret = {k: v.avg for k, v in avg_loss_stats.items()}

        # # randomly do multi-scaling for dataset every epoch
        # data_loader.dataset.rand_scale()  # re-assign scale for each batch
        #
        # # shuffule the dataset every epoch
        # data_loader.dataset.shuffle()  # re-assign file id for each idx

        bar.finish()
        ret['time'] = bar.elapsed_td.total_seconds() / 60.0

        return ret

    def debug(self, batch, output, iter_id):
        raise NotImplementedError

    def save_result(self, output, batch, results):
        raise NotImplementedError

    # def _get_losses(self, opt):
    #     raise NotImplementedError

    def val(self, epoch, data_loader):
        return self.run_epoch('val', epoch, data_loader)

    def train(self, epoch, data_loader):
        return self.run_epoch('train', epoch, data_loader)

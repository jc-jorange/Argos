import time
import traceback
from enum import Enum, unique
import cv2
import os
import numpy as np
import torch

from lib.multiprocess import ProducerProcess
from lib.multiprocess.SharedMemory import E_ProducerOutputName_Indi
from lib.input_data_loader import EInputDataType, loader_factory, BaseInputDataLoader


@unique
class EImageInfo(Enum):
    Data = 1
    Size = 2


class ImageLoaderProcess(ProducerProcess):
    prefix = 'Argus-SubProcess-ImageLoader_'
    dir_name = 'input'
    log_name = 'Image_Loader_Log'

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(ImageLoaderProcess, self).__init__(*args, **kwargs)

        self.frame_dir = None if self.opt.output_format == 'text' \
            else self.making_dir(self.results_save_dir, self.opt.frame_dir)

        self.data_loader = None

        self.path = self.opt.input_path[self.idx]
        self.loader_mode = self.opt.input_mode

        self.load_time = 0.0

    def run_begin(self) -> None:
        super(ImageLoaderProcess, self).run_begin()

        self.logger.info("Start Creating Input Dataloader")
        if self.loader_mode == EInputDataType.Image:
            self.logger.info(f'Start Loading Images in {self.path}')
        if self.loader_mode == EInputDataType.Video:
            self.logger.info(f'Start Loading Video in {self.path}')
        if self.loader_mode == EInputDataType.Address:
            self.logger.info(f'Start Loading From Camera in {self.path}')
        self.data_loader: BaseInputDataLoader = loader_factory[self.loader_mode](self.path, self.opt.net_input_shape)

    def run_action(self) -> None:
        self.logger.info("Start loading images")
        start_time = time.perf_counter()

        hub_image_data = self.producer_result_hub.output[E_ProducerOutputName_Indi.ImageData]
        hub_image_origin_shape = self.producer_result_hub.output[E_ProducerOutputName_Indi.ImageOriginShape]
        hub_frame_id = self.producer_result_hub.output[E_ProducerOutputName_Indi.FrameID]
        hub_b_loading = self.producer_result_hub.output[E_ProducerOutputName_Indi.bInputLoading]

        try:
            for path, img, img_0 in self.data_loader:
                if not self.opt.realtime:
                    while hub_image_data.qsize() > self.opt.load_buffer:
                        pass
                    img_send = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)
                    hub_image_data.put((self.data_loader.count, img_send, img_0.shape))
                    del img_send
                else:
                    hub_image_data[:] = img[:]

                hub_frame_id.value = self.data_loader.count
                hub_image_origin_shape[:] = img_0.shape[:]

                if self.frame_dir:
                    cv2.imwrite(
                        os.path.join(self.frame_dir, '{:05d}.jpg'.format(self.data_loader.count)),
                        img_0
                    )

                del img

        except:
            traceback.print_exc()
            pass
        if not self.opt.realtime:
            while not hub_image_data.empty():
                pass

        hub_b_loading.value = False

        end_time = time.perf_counter()
        self.load_time = end_time - start_time

    def run_end(self) -> None:
        self.logger.info(
            f"Total receive {self.data_loader.count} frames in {self.load_time} s"
        )

        super().run_end()

        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)

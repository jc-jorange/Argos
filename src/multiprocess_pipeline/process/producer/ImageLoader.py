import time
import traceback
from enum import Enum, unique
import cv2
import os
import torch

from ._masterclass import ProducerProcess
from src.multiprocess_pipeline.workers.image_loader import E_ImageLoaderName, factory_image_loader, BaseImageLoader
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_PipelineSharedDataName, E_SharedDataFormat


@unique
class EImageInfo(Enum):
    Data = 1
    Size = 2


class ImageLoaderProcess(ProducerProcess):
    prefix = 'Argos-SubProcess-ImageLoader_'
    dir_name = 'image_input'
    log_name = 'Image_Loader_Log'

    shared_data = {
        E_PipelineSharedDataName.Image.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1, ),
            E_SharedDataFormat.data_value.name: 0
        },
    }

    def __init__(self,
                 image_path: str,
                 loader: str,
                 normalized_image_shape: list,
                 *args,
                 timestamp_path='',
                 load_buffer=8,
                 show_image=False,
                 **kwargs,
                 ):
        super(ImageLoaderProcess, self).__init__(*args, **kwargs)

        self.data_loader = None
        self.loader_name = loader
        self.timestamp_path = timestamp_path
        self.image_path = image_path
        self.load_buffer = max(load_buffer, 1)
        self.normalized_image_shape = tuple(normalized_image_shape)
        self.bshow_image = show_image

        if self.loader_name not in E_ImageLoaderName.__members__.keys():
            raise KeyError(f'loader {self.loader_name} is not a valid loader')

        self.count = 0
        self.load_time = 0.0
        self.fps_send = 0
        self.fps_read = 0
        self.fps_avg = 0

    def run_begin(self) -> None:
        super(ImageLoaderProcess, self).run_begin()
        if os.path.isfile(self.timestamp_path):
            self.logger.info(f'Reading timestamp @ {self.timestamp_path}')
        else:
            self.logger.info(f'Reading timestamp dynamically')

        self.logger.info(f"Start Creating Image Dataloader @ {self.image_path}")
        self.data_loader: BaseImageLoader = \
            factory_image_loader[self.loader_name](
                self.image_path,
                self.normalized_image_shape,
                timestamp_path=self.timestamp_path)

        t1 = time.perf_counter()
        while not self.data_loader.pre_process():
            t2 = time.perf_counter()
            dt = t2 - t1
            if dt >= 5:
                self.logger.info(f'Waiting read image @ {self.image_path}')
                t1 = t2
        self.logger.info(f'Finish initial read image @ {self.image_path}')

    def run_action(self) -> None:
        self.logger.info("Start loading images")
        start_time = time.perf_counter()

        hub_image = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.Image.name]

        try:
            t_read_start = time.perf_counter()

            for timestamp, path, img_0, img in self.data_loader:
                t_read_end = time.perf_counter()

                if len(self.data_loader) > 1:
                    total_image = len(self.data_loader)
                    self.logger.info(f'loading image: ['
                                     f'{(self.count / total_image):.2%}, '
                                     f'{self.count}/{total_image}, '
                                     f'{self.fps_avg:.2f} image/s'
                                     f']')

                self.count += 1
                self.logger.debug(f'Read Img {int(self.data_loader.count)} from {path} @ timestamp {timestamp}')
                self.logger.debug(f'delta time as image loader get: {self.get_current_timestamp() - timestamp}')

                while hub_image.size() > self.load_buffer:
                    if not self.opt.realtime:
                        pass
                    else:
                        tmp = hub_image.get()
                        del tmp
                        continue

                if self.bshow_image and self.opt.allow_show_image:
                    cv2.imshow('Image Loader ' + self.pipeline_name, img_0)
                    cv2.waitKey(1)

                t_send_start = time.perf_counter()

                img = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)
                img.share_memory_()
                if self.opt.half_precision:
                    img = img.type(torch.HalfTensor).to(self.opt.device)
                frame_id = int(self.data_loader.count)
                img_shape = img_0.shape

                image_data = (timestamp, frame_id, img_shape, img_0, img)  # frame, timestamp, original shape, img

                hub_image.set(image_data)

                t_send_end = time.perf_counter()
                self.logger.debug(f'delta time as image loader send: {self.get_current_timestamp() - timestamp}')

                dt_send = t_send_end - t_send_start
                dt_read = t_read_end - t_read_start
                dt_all = t_send_end - start_time
                self.fps_send = 1 / dt_send
                self.fps_read = 1 / dt_read
                self.fps_avg = self.count / dt_all

                if self.count % 10 == 0 and self.count != 0:
                    self.logger.info(
                        f'Reading frame count {self.count}: '
                        f'average: {self.fps_avg:.2f} dps, '
                        f'current read only: {self.fps_read:.2f} dps; '
                        f'current send only: {self.fps_send:.2f} dps; '
                    )

                cv2.imwrite(
                    os.path.join(self.results_save_dir, '{:05d}.jpg'.format(self.data_loader.count)),
                    img_0
                )
                self.logger.debug(f'Save img {int(self.data_loader.count)} from {path}')
                t_read_start = time.perf_counter()

        except:
            traceback.print_exc()
            pass

        end_time = time.perf_counter()
        self.load_time = end_time - start_time

    def run_end(self) -> None:
        self.logger.info(
            f"Total receive {self.data_loader.count} frames in {self.load_time} s"
        )

        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)

        super(ImageLoaderProcess, self).run_end()

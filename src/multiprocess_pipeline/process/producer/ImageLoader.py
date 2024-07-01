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
        E_PipelineSharedDataName.ImageData.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.FrameID.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.ImageOriginShape.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.SharedArray_Int,
            E_SharedDataFormat.data_shape.name: (3,),
            E_SharedDataFormat.data_value.name: 0
        },

        E_PipelineSharedDataName.ImageTimestamp.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
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
                 **kwargs,
                 ):
        super(ImageLoaderProcess, self).__init__(*args, **kwargs)

        self.data_loader = None
        self.loader_name = loader
        self.timestamp_path = timestamp_path
        self.image_path = image_path
        self.load_buffer = load_buffer
        self.normalized_image_shape = tuple(normalized_image_shape)

        if self.loader_name not in E_ImageLoaderName.__members__.keys():
            raise KeyError(f'loader {self.loader_name} is not a valid loader')

        self.count = 0
        self.load_time = 0.0
        self.fps_cur = 0
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
                self.logger.info(f'Waiting read @ {self.image_path}')
                t1 = t2

    def run_action(self) -> None:
        self.logger.info("Start loading images")
        start_time = time.perf_counter()

        hub_image_data = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.ImageData.name]
        hub_image_origin_shape = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.ImageOriginShape.name]
        hub_frame_id = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.FrameID.name]
        hub_timestamp = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.ImageTimestamp.name]

        try:
            for timestamp, path, img_0, img in self.data_loader:
                t_each_start = time.perf_counter()
                self.count += 1
                self.logger.debug(f'Read Img {int(self.data_loader.count)} from {path}')
                # if not self.opt.realtime:
                #     while hub_image_data.size() > self.load_buffer:
                #         # hub_frame_id.get()
                #         # hub_image_origin_shape.get()
                #         # hub_image_data.get()
                #         # hub_timestamp.get()
                #         pass
                img = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)

                hub_frame_id.set(int(self.data_loader.count))
                hub_image_origin_shape.set(img_0.shape[:])
                hub_image_data.set(img)
                hub_timestamp.set(timestamp)
                self.logger.debug(f'Set Img and timestamp')

                t_each_end = time.perf_counter()
                dt_each = t_each_end - t_each_start
                dt_all = t_each_end - start_time
                self.fps_cur = 1 / dt_each
                self.fps_avg = self.count / dt_all

                if self.count % 10 == 0 and self.count != 0:
                    self.logger.info(
                        f'Reading frame count {self.count}: '
                        f'average dps: {self.fps_avg:.2f}, '
                        f'current dps: {self.fps_cur:.2f}; '
                    )

                cv2.imwrite(
                    os.path.join(self.results_save_dir, '{:05d}.jpg'.format(self.data_loader.count)),
                    img_0
                )
                self.logger.debug(f'Save img {int(self.data_loader.count)} from {path}')

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

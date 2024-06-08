import time
import traceback
from enum import Enum, unique
import cv2
import os
import torch

from ..Image import ImageProcess_Master
from lib.multiprocess_pipeline.workers.image_loader import E_ImageLoaderName, factory_image_loader, BaseImageLoader
from lib.multiprocess_pipeline.process.SharedDataName import E_PipelineSharedDataName


@unique
class EImageInfo(Enum):
    Data = 1
    Size = 2


class ImageLoaderProcess(ImageProcess_Master):
    prefix = 'Argus-SubProcess-ImageLoader_'
    dir_name = 'image_input'
    log_name = 'Image_Loader_Log'

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

        self.load_time = 0.0

    def run_begin(self) -> None:
        super(ImageLoaderProcess, self).run_begin()

        self.logger.info(f"Start Creating Image Dataloader @ {self.image_path}")

        if os.path.isfile(self.timestamp_path):
            self.logger.info(f'Reading timestamp @ {self.timestamp_path}')
        else:
            self.logger.info(f'Reading timestamp dynamically')

        self.logger.info(f'Waiting read @ {self.image_path}')
        self.data_loader: BaseImageLoader = \
            factory_image_loader[self.loader_name](
                self.image_path,
                self.normalized_image_shape,
                timestamp_path=self.timestamp_path)

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
                if not self.opt.realtime:
                    while hub_image_data.size()[0] > self.load_buffer and \
                            self.data_hub.dict_bLoadingFlag[self.pipeline_name].value:
                        pass
                img = torch.from_numpy(img).unsqueeze(0).to(self.opt.device)

                hub_frame_id.set(int(self.data_loader.count))
                hub_image_origin_shape.set(img_0.shape[:])
                hub_image_data.set(img)
                hub_timestamp.set(timestamp)

                cv2.imwrite(
                    os.path.join(self.results_save_dir, '{:05d}.jpg'.format(self.data_loader.count)),
                    img_0
                )

                cv2.imshow('test', img_0)
                cv2.waitKey(1)

        except:
            traceback.print_exc()
            pass

        end_time = time.perf_counter()
        self.load_time = end_time - start_time

    def run_end(self) -> None:
        self.logger.info(
            f"Total receive {self.data_loader.count} frames in {self.load_time} s"
        )

        hub_image_data = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.ImageData.name]
        hub_image_origin_shape = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.ImageOriginShape.name]
        hub_frame_id = self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.FrameID.name]
        if not self.opt.realtime:
            while hub_image_data.size()[0] > 0:
                try:
                    hub_image_data.get()
                    hub_image_origin_shape.get()
                    hub_frame_id.get()
                except RuntimeError:
                    pass

        super().run_end()
        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)
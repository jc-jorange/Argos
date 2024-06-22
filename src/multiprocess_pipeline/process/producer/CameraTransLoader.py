import time
import inspect

from ._masterclass import ProducerProcess
from src.multiprocess_pipeline.workers.camera_trans_loader import factory_camera_trans_loader, E_CameraTransLoaderName
from src.multiprocess_pipeline.shared_structure import E_SharedSaveType, E_PipelineSharedDataName, E_SharedDataFormat


class CameraTransLoaderProcess(ProducerProcess):
    prefix = 'Argus-SubProcess-CameraTransLoader_'
    dir_name = 'camera_trans_input'
    log_name = 'Camera_Trans_Loader_Log'

    shared_data = {
        E_PipelineSharedDataName.CameraTransform.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
        E_PipelineSharedDataName.TransformTimestamp.name: {
            E_SharedDataFormat.data_type.name: E_SharedSaveType.Queue,
            E_SharedDataFormat.data_shape.name: (1,),
            E_SharedDataFormat.data_value.name: 0
        },
    }

    def __init__(self,
                 loader: str,
                 source,
                 *args,
                 load_buffer=8,
                 **kwargs,
                 ):

        if loader not in E_CameraTransLoaderName.__members__.keys():
            raise KeyError(f'loader {loader} is not a valid loader')
        self.loader = factory_camera_trans_loader[loader]

        loader_kwargs = {}
        tmp = inspect.signature(self.loader).bind(source, **kwargs)
        tmp.apply_defaults()
        tmp_keys = tmp.arguments.keys()
        for k in tmp_keys:
            if k in kwargs.keys():
                loader_kwargs[k] = kwargs.pop(k)

        super(CameraTransLoaderProcess, self).__init__(*args, **kwargs)

        self.source = source
        self.load_buffer = load_buffer
        self.loader_kwargs = loader_kwargs

        self.count = 0
        self.load_time = 0
        self.dps_avg = 0
        self.dps_cur = 0

    def run_begin(self) -> None:
        super(CameraTransLoaderProcess, self).run_begin()

        self.logger.info(f"Start Creating Camera Transform Loader @ "
                         f"{self.source if isinstance(self.source, str) else 'Fixed Camera'}")
        self.logger.info(f'Waiting read @ {self.source}')
        self.loader = self.loader(self.source, **self.loader_kwargs)

    def run_action(self) -> None:
        self.logger.info("Start loading camera transform")

        hub_camera_trans = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.CameraTransform.name]
        hub_camera_timestamp = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.TransformTimestamp.name]

        start_time = time.perf_counter()
        for timestamp, path, trans in self.loader:
            if timestamp and trans:
                each_frame_start_time = time.perf_counter()
                self.count += 1
                self.logger.debug(f'Read Camera Transform {self.count} from {path}')

                hub_camera_timestamp.set(timestamp)
                hub_camera_trans.set(trans)
                self.logger.debug(f'Set Camera Transform and timestamp')

                each_frame_end_time = time.perf_counter()
                delta_time_each = each_frame_end_time - each_frame_start_time
                delta_time_all = each_frame_end_time - start_time
                self.dps_avg = self.count / delta_time_all
                self.dps_cur = 1 / delta_time_each

                if self.count % 50 == 0 and self.count != 0:
                    self.logger.info(
                        f'Reading Data count {self.count}: '
                        f'average dps: {self.dps_avg:.2f}, '
                        f'current dps: {self.dps_cur:.2f}; '
                    )

        end_time = time.perf_counter()
        self.load_time = end_time - start_time

    def run_end(self) -> None:
        self.logger.info(
            f"Total receive {self.loader.count} transform data in {self.load_time} s"
        )

        hub_camera_trans = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.CameraTransform.name]
        hub_camera_timestamp = \
            self.data_hub.dict_shared_data[self.pipeline_name][E_PipelineSharedDataName.TransformTimestamp.name]

        if not self.opt.realtime:
            for i in range(hub_camera_trans.size()):
                try:
                    hub_camera_trans.get()
                    hub_camera_timestamp.get()
                except RuntimeError:
                    pass

        super().run_end()
        self.logger.info('-' * 5 + 'Camera Transform Receiver Finished' + '-' * 5)

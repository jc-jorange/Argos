import time
from enum import Enum, unique

import numpy as np

from ..multiprocess import BaseProcess, ESharedDictType
from lib.input_data_loader import EInputDataType, loader_factory
import lib.multiprocess.Shared as Sh
from multiprocessing import Queue


@unique
class EImageInfo(Enum):
    Data = 1
    Size = 2


class ImageLoaderProcess(BaseProcess):
    prefix = 'Argus-SubProcess-ImageLoader_'

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(ImageLoaderProcess, self).__init__(*args, **kwargs)
        self.making_process_main_save_dir('camera_raw_')

        self.path = self.opt.input_path[self.idx]
        self.loader_mode = self.opt.input_mode
        self.data_loader = None

        self.load_time = 0.0

    # def read_image(self):
    #     img = None
    #     bReadResult = False
    #     start_time = time.perf_counter()
    #
    #     if self.connect_type == 'UDP':
    #         self.ConnectionSocket.setblocking(False)
    #         self.ConnectionSocket.settimeout(1)
    #         try:
    #             clip_num = 64000
    #             flag_data, address = self.ConnectionSocket.recvfrom(4 + 4 + 4 + 8)
    #             if flag_data and "I///" in flag_data.decode("utf-8", "ignore"):
    #                 flag = flag_data
    #                 self.width = int(flag[4:8])
    #                 self.height = int(flag[8:12])
    #                 data_count_total = int(flag[12:20])
    #                 data_count = 0
    #                 img_bytes = b''
    #
    #                 if data_count_total > 0:
    #                     total_count = data_count_total // clip_num
    #                     left = data_count_total % clip_num
    #                     for i in range(total_count):
    #                         data, address = self.ConnectionSocket.recvfrom(clip_num)
    #                         img_bytes += data
    #                         data_count += len(data)
    #                     if left > 0:
    #                         data, address = self.ConnectionSocket.recvfrom(left)
    #                         img_bytes += data
    #                         data_count += len(data)
    #
    #                     try:
    #                         img = np.asarray(bytearray(img_bytes))
    #                         img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
    #                         img = create_gamma_img(3, img)
    #                         # self.img_0 = img
    #                         self.logger.info(
    #                             "UDP receive image at {}".format(datetime.datetime.now()))
    #                         bReadResult = True
    #                     except:
    #                         pass
    #         except timeout:
    #             self.keep_process = False
    #         except OSError:
    #             self.keep_process = False
    #             pass
    #
    #     elif self.connect_type == 'TCP':
    #         # self.ConnectionSocket.setblocking(False)
    #         # clear_socket_buffer(self.ConnectionSocket, 1)
    #         # self.ConnectionSocket.setblocking(True)
    #         try:
    #             data = self.ConnectionSocket.recv(4)
    #             # print(data.decode("utf-8", "ignore"))
    #             if b'\x00\x00' in data:
    #                 # self.keep_process = False
    #                 pass
    #             if data and data.decode("utf-8", "ignore") == "I///":
    #                 start_time = time.perf_counter()
    #                 flag_data = self.ConnectionSocket.recv(4+4+8)
    #                 flag = flag_data.decode("utf-8", "ignore")
    #                 self.width = int(flag[0:4])
    #                 self.height = int(flag[4:8])
    #                 data_count_total = int(flag[8:16])
    #                 data_count = 0
    #                 img_bytes = b''
    #
    #                 if data_count_total > 0:
    #                     while data_count < data_count_total:
    #                         data = self.ConnectionSocket.recv(data_count_total)
    #                         img_bytes += data
    #                         data_count += len(data)
    #
    #                     img = np.asarray(bytearray(img_bytes))
    #                     img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
    #                     img = create_gamma_img(1.0, img)
    #                     # self.img_0 = img
    #                     # self.logger.info(
    #                     #     "TCP receive image at {}".format(datetime.datetime.now()))
    #                     bReadResult = True
    #                 end_time = time.perf_counter()
    #                 # print('only read image: {}'.format(end_time - start_time))
    #             elif not data:
    #                 self.keep_process = False
    #         except Exception:
    #             traceback.print_exc()
    #             self.keep_process = False
    #
    #     self.total_frames += 1
    #
    #     end_time = time.perf_counter()
    #     recv_time = end_time - start_time
    #     self.logger.debug("Receive {} frame in time {} s".format(self.total_frames, recv_time))
    #     return img, bReadResult

    def run_begin(self) -> None:
        super(ImageLoaderProcess, self).run_begin()
        self.set_logger_file_handler(self.name, self.main_output_dir)
        self.logger.info(f"This is the Image Receiver Process No.{self.idx}")

        self.logger.info("Start Creating Input Dataloader")
        if self.loader_mode == EInputDataType.Image:
            self.logger.info(f'Start Loading Images in {self.path}')
        if self.loader_mode == EInputDataType.Video:
            self.logger.info(f'Start Loading Video in {self.path}')
        if self.loader_mode == EInputDataType.Address:
            self.logger.info(f'Start Loading From Camera in {self.path}')
        self.data_loader = loader_factory[self.loader_mode](self.path)

    def run_action(self) -> None:
        self.logger.info("Start loading images")
        start_time = time.perf_counter()

        for path, img, img_0 in self.data_loader:
            if path:
                self.container_shared_dict[ESharedDictType.Image_Input_List][self.idx].append((path, img, img_0))

        end_time = time.perf_counter()
        self.load_time = end_time - start_time

    def run_end(self) -> None:
        super().run_end()
        self.logger.info(
            f"Total receive {self.data_loader.count} frames in {self.load_time} s"
        )

        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)

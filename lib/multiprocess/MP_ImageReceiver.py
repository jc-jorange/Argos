import os
import time
import datetime
from socket import *
from enum import Enum, unique
import traceback
import numpy as np
import cv2

from ..multiprocess import BaseProcess, ESharedDictType, EMultiprocess
from lib.dataset.utils.utils import create_gamma_img, clear_socket_buffer


@unique
class ESupportConnectionType(Enum):
    TCP = 1
    UDP = 2


@unique
class EImageInfo(Enum):
    Data = 1
    Size = 2


class ImageReceiverProcess(BaseProcess):
    prefix = 'Argus-SubProcess-ImageReceiver_'

    def __init__(self,
                 *args,
                 **kwargs,
                 ):
        super(ImageReceiverProcess, self).__init__(*args, **kwargs)

        self.making_process_main_save_dir('camera_raw_')

        path = self.opt.input_path[self.idx]
        address_str_list = path.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        try:
            ESupportConnectionType[self.connect_type]
        except KeyError:
            self.logger.warn("None support connection type {:s}. Please check it.".format(self.connect_type))
            self.close()

        self.address = (ip, port)

        self.keep_process = True
        self.img_0 = None
        self.height = 0
        self.width = 0

        self.total_frames = 0

    def read_image(self):
        img = None
        bReadResult = False
        start_time = time.perf_counter()

        if self.connect_type == 'UDP':
            self.ConnectionSocket.setblocking(False)
            self.ConnectionSocket.settimeout(1)
            try:
                clip_num = 64000
                flag_data, address = self.ConnectionSocket.recvfrom(4 + 4 + 4 + 8)
                if flag_data and "I///" in flag_data.decode("utf-8", "ignore"):
                    flag = flag_data
                    self.width = int(flag[4:8])
                    self.height = int(flag[8:12])
                    data_count_total = int(flag[12:20])
                    data_count = 0
                    img_bytes = b''

                    if data_count_total > 0:
                        total_count = data_count_total // clip_num
                        left = data_count_total % clip_num
                        for i in range(total_count):
                            data, address = self.ConnectionSocket.recvfrom(clip_num)
                            img_bytes += data
                            data_count += len(data)
                        if left > 0:
                            data, address = self.ConnectionSocket.recvfrom(left)
                            img_bytes += data
                            data_count += len(data)

                        try:
                            img = np.asarray(bytearray(img_bytes))
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
                            img = create_gamma_img(3, img)
                            # self.img_0 = img
                            self.logger.info(
                                "UDP receive image at {}".format(datetime.datetime.now()))
                            bReadResult = True
                        except:
                            pass
            except timeout:
                self.keep_process = False
            except OSError:
                self.keep_process = False
                pass

        elif self.connect_type == 'TCP':
            # self.ConnectionSocket.setblocking(False)
            # clear_socket_buffer(self.ConnectionSocket, 1)
            # self.ConnectionSocket.setblocking(True)
            try:
                data = self.ConnectionSocket.recv(4)
                # print(data.decode("utf-8", "ignore"))
                if b'\x00\x00' in data:
                    # self.keep_process = False
                    pass
                if data and data.decode("utf-8", "ignore") == "I///":
                    start_time = time.perf_counter()
                    flag_data = self.ConnectionSocket.recv(4+4+8)
                    flag = flag_data.decode("utf-8", "ignore")
                    self.width = int(flag[0:4])
                    self.height = int(flag[4:8])
                    data_count_total = int(flag[8:16])
                    data_count = 0
                    img_bytes = b''

                    if data_count_total > 0:
                        while data_count < data_count_total:
                            data = self.ConnectionSocket.recv(data_count_total)
                            img_bytes += data
                            data_count += len(data)

                        img = np.asarray(bytearray(img_bytes))
                        img = cv2.imdecode(img, cv2.IMREAD_COLOR) # BGR
                        img = create_gamma_img(1.0, img)
                        # self.img_0 = img
                        # self.logger.info(
                        #     "TCP receive image at {}".format(datetime.datetime.now()))
                        bReadResult = True
                    end_time = time.perf_counter()
                    # print('only read image: {}'.format(end_time - start_time))
                elif not data:
                    self.keep_process = False
            except Exception:
                traceback.print_exc()
                self.keep_process = False

        self.total_frames += 1

        end_time = time.perf_counter()
        recv_time = end_time - start_time
        self.logger.debug("Receive {} frame in time {} s".format(self.total_frames, recv_time))
        return img, bReadResult

    def run(self):
        super(ImageReceiverProcess, self).run()
        self.set_logger_file_handler(self.name, self.main_output_dir)
        self.logger.info("This is the Image Receiver Process No.{:d}".format(self.idx))

        if self.connect_type == ESupportConnectionType.TCP.name:
            ServerSocket = socket(AF_INET, SOCK_STREAM)
            ServerSocket.bind(self.address)
            self.logger.info("Waiting for TCP connection at {}:{}".format(self.address[0], self.address[1]))
            ServerSocket.listen()

            self.ConnectionSocket, address = ServerSocket.accept()
            self.logger.info(
                "Successfully connected to {} and connection on {}".format(self.address, address)
            )
        elif self.connect_type == ESupportConnectionType.UDP.name:
            ServerSocket = socket(AF_INET, SOCK_DGRAM)
            ServerSocket.bind(self.address)
            self.ConnectionSocket = ServerSocket
            self.logger.info("UDP connection at {}:{}".format(self.address[0], self.address[1]))
        else:
            raise

        self.ConnectionSocket.settimeout(60)

        self.logger.info("Waiting connection send image at {}:{}".format(self.address[0], self.address[1]))
        start_time = time.perf_counter()
        bDoOnce = True
        while self.keep_process:
            img_0, bRecv = self.read_image()
            if bRecv:
                img_0.flatten()
                self.container_shared_dict[ESharedDictType.Image][self.idx].set_data(EImageInfo.Data, img_0)
                self.container_shared_dict[ESharedDictType.Image][self.idx].set_data(EImageInfo.Size, img_0.size)
                if bDoOnce:
                    self.logger.info("Receive initial image at {}:{}".format(self.address[0], self.address[1]))
                    bDoOnce = False

        end_time = time.perf_counter()
        self.logger.info(
            "Total receive {} frames in {} s at {}:{}".format(
                self.total_frames, end_time-start_time, self.address[0], self.address[1]
            )
        )

        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)

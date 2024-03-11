import os
import time
import datetime
from socket import *
from enum import Enum, unique
import traceback
import numpy as np
import cv2

from collections import defaultdict
from multiprocessing import Process, Lock, Value, shared_memory

from lib.utils.logger import logger
from lib.tracker.utils.utils import mkdir_if_missing
from lib.dataset.utils.utils import create_gamma_img, clear_socket_buffer
import lib.multiprocess.Shared as Sh


@unique
class E_SupportConnectionType(Enum):
    TCP = 1
    UDP = 2


@unique
class E_ImageInfo(Enum):
    Data = 1
    Size = 2


class ImageReceiver(Process):
    def __init__(self,
                 idx: int,
                 opt,
                 shared_image_dict: Sh.SharedDict
                 ):
        super().__init__()

        self.idx = idx
        self.opt = opt
        self.shared_image_dict = shared_image_dict

        self.name = 'Argus-SubProcess-ImageReceiver_' + str(idx)
        self.output_dir = os.path.join(opt.save_dir, 'camera_raw_' + str(idx+1))
        mkdir_if_missing(self.output_dir)

        self.logger = logger.add_logger(os.getpid())
        logger.add_stream_handler(os.getpid())
        logger.add_file_handler(os.getpid(), self.name, self.output_dir)

        logger.set_logger_level(os.getpid(), 'debug' if self.opt.debug else 'info')
        self.logger.info("This is the Image Receiver Process No.{:d}".format(self.idx))

        path = opt.input_path[idx]
        address_str_list = path.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        try:
            E_SupportConnectionType[self.connect_type]
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
        if self.connect_type == E_SupportConnectionType.TCP.name:
            ServerSocket = socket(AF_INET, SOCK_STREAM)
            ServerSocket.bind(self.address)
            self.logger.info("Waiting for TCP connection at {}:{}".format(self.address[0], self.address[1]))
            ServerSocket.listen()

            self.ConnectionSocket, address = ServerSocket.accept()
            self.logger.info(
                "Successfully connected to {} and connection on {}".format(self.address, address)
            )
        elif self.connect_type == E_SupportConnectionType.UDP.name:
            ServerSocket = socket(AF_INET, SOCK_DGRAM)
            ServerSocket.bind(self.address)
            self.ConnectionSocket = ServerSocket
            self.logger.info("UDP connection at {}:{}".format(self.address[0], self.address[1]))
        else:
            raise

        self.ConnectionSocket.settimeout(60)

        self.logger.info("Waiting connection send image at {}:{}".format(self.address[0], self.address[1]))
        bInitialImg = False
        while not bInitialImg:
            ini_img, bInitialImg = self.read_image()
        self.shared_image_dict.set_data(E_ImageInfo.Data, ini_img)
        self.shared_image_dict.set_data(E_ImageInfo.Size, ini_img.size)
        self.logger.info("Receive initial image at {}:{}".format(self.address[0], self.address[1]))

        start_time = time.perf_counter()
        while self.keep_process:
            img_0, bRecv = self.read_image()
            if bRecv:
                img_0.flatten()
                self.shared_image_dict.set_data(E_ImageInfo.Data, img_0)

        end_time = time.perf_counter()
        self.logger.info(
            "Total receive {} frames in {} s at {}:{}".format(
                self.total_frames, end_time-start_time, self.address[0], self.address[1]
            )
        )

        self.logger.info('-' * 5 + 'Image Receiver Finished' + '-' * 5)

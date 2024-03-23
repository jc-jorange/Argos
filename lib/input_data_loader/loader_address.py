import cv2
import numpy as np
import traceback
from socket import *
from enum import Enum, unique

from lib.input_data_loader import BaseInputDataLoader
from lib.input_data_loader.utils import create_gamma_img


@unique
class ESupportConnectionType(Enum):
    TCP = 1
    UDP = 2


class AddressDataLoader(BaseInputDataLoader):
    def __init__(self, *args):
        super(AddressDataLoader, self).__init__(*args)
        address_str_list = self.data_path.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        assert ESupportConnectionType[self.connect_type], \
            f'None support connection type {self.connect_type}. Please check it.'
        self.address = (ip, port)
        self.len = 1

        self.ConnectionSocket = None
        self.b_socket_alive = True

        self.connect()

    def connect(self):
        if self.connect_type == ESupportConnectionType.TCP.name:
            server_socket = socket(AF_INET, SOCK_STREAM)
            server_socket.bind(self.address)
            server_socket.listen()

            self.ConnectionSocket, address = server_socket.accept()
        elif self.connect_type == ESupportConnectionType.UDP.name:
            server_socket = socket(AF_INET, SOCK_DGRAM)
            server_socket.bind(self.address)
            self.ConnectionSocket = server_socket
        else:
            raise

        self.ConnectionSocket.settimeout(60)

    def __next__(self):
        super(AddressDataLoader, self).__next__()
        if not self.b_socket_alive:
            raise StopIteration
        return self.read_image(self.count)

    def read_action(self, idx) -> (str, np.ndarray):
        super(AddressDataLoader, self).read_action(idx)
        img_path = self.data_path
        img = None

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
                            img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR
                            img = create_gamma_img(2.2, img)
                        except:
                            pass
            except timeout or OSError:
                self.b_socket_alive = False

        elif self.connect_type == 'TCP':
            try:
                data = self.ConnectionSocket.recv(4)
                if b'\x00\x00' in data:
                    pass
                if data and data.decode("utf-8", "ignore") == "I///":
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

                elif not data:
                    self.b_socket_alive = False
            except Exception:
                traceback.print_exc()
                self.b_socket_alive = False
        assert img is not None, 'Failed to load frame ' + img_path
        self.image_shape = img.shape
        return img_path, img

import time

import cv2
import numpy as np
import struct
from socket import *
from enum import Enum, unique

from ._masterclass import BaseImageLoader
from .utils import create_gamma_img


FLAG_HEAD = 'I///'
_udp_clip_num = 51200

_bytes_num_flag_head = len(FLAG_HEAD)
_bytes_num_timestamp = len(struct.pack('q', 0))
_bytes_num_image_width = len(struct.pack('I', 0))
_bytes_num_image_height = len(struct.pack('I', 0))
_bytes_num_data_count = len(struct.pack('I', 0))

_total_head_num = _bytes_num_flag_head + \
                  _bytes_num_timestamp + \
                  _bytes_num_image_width + \
                  _bytes_num_image_height + \
                  _bytes_num_data_count


@unique
class E_SupportConnectionType(Enum):
    TCP = 1
    UDP = 2


class AddressImageLoader(BaseImageLoader):
    def __init__(self, *args, **kwargs):
        super(AddressImageLoader, self).__init__(*args, **kwargs)
        address_str_list = self.data_path.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        assert E_SupportConnectionType[self.connect_type], \
            f'None support connection type {self.connect_type}. Please check it.'
        self.address = (ip, port)
        self.len = 1

        self.ConnectionSocket = None
        self.b_socket_alive = True

        self.width = 0
        self.height = 0
        self.recv_timestamp = 0
        self.data_count_total = 0

        self.misread_count = 0

    def pre_process(self) -> bool:
        super(AddressImageLoader, self).pre_process()
        if self.ConnectionSocket:
            pass
        else:
            self.connect()

        self.b_socket_alive = True
        first_data = self.read_action(0)[0]
        if first_data:
            b_has_data = True
            while b_has_data:
                b_has_data = self.flush_listen()

            return True
        else:
            return False

    def connect(self) -> None:
        if self.connect_type == E_SupportConnectionType.TCP.name:
            server_socket = socket(AF_INET, SOCK_STREAM)
            server_socket.bind(self.address)
            server_socket.listen()

            self.ConnectionSocket, address = server_socket.accept()
        elif self.connect_type == E_SupportConnectionType.UDP.name:
            server_socket = socket(AF_INET, SOCK_DGRAM)
            udp_address = ('', self.address[1])
            if self.address[0] in ('127.0.0.1', 'localhost'):
                udp_address = ('localhost', self.address[1])
            server_socket.bind(udp_address)
            self.ConnectionSocket = server_socket
        else:
            raise

        self.ConnectionSocket.setblocking(False)
        self.ConnectionSocket.settimeout(60)

    def read_action(self, idx) -> (int, str, np.ndarray):
        super(AddressImageLoader, self).read_action(idx)

        self.ConnectionSocket.settimeout(1)

        img = None

        while self.b_socket_alive:
            try:
                if self.bWith_Flag:
                    if not self.recv_flag():
                        continue
                self.recv_timestamp, img = self.recv_data()
                if isinstance(img, np.ndarray):
                    return self.recv_timestamp, self.data_path, img

            except timeout:
                self.b_socket_alive = False

            except OSError:
                pass

        return self.recv_timestamp, self.data_path, img

    def flush_listen(self) -> bool:
        self.ConnectionSocket.settimeout(0)
        while True:
            try:
                data = self.ConnectionSocket.recv(51200)
                if len(data) < 51200:
                    return False
            except:
                return False

    def recv_flag(self) -> bool:
        flag_data = self.ConnectionSocket.recv(_udp_clip_num)
        if flag_data:
            if len(flag_data) != _total_head_num:
                return False
            flag_head, self.recv_timestamp, self.width, self.height, self.data_count_total = \
                struct.unpack(f'={_bytes_num_flag_head}sQIII', flag_data)
            flag_head = flag_head.decode("utf-8", "ignore")
            return flag_head and flag_head == FLAG_HEAD
        else:
            return False

    def recv_data(self):
        data_count = 0
        img_bytes = b''
        recv_timestamp = int(time.time() * 1000)
        total_count_tmp = (self.data_count_total+8) if self.bWith_Flag else self.data_count_total

        if self.connect_type == E_SupportConnectionType.TCP.name:
            while data_count < total_count_tmp:
                data = self.ConnectionSocket.recv(total_count_tmp)
                img_bytes += data
                data_count += len(data)

        elif self.connect_type == E_SupportConnectionType.UDP.name:
            total_count = total_count_tmp // _udp_clip_num
            left = total_count_tmp % _udp_clip_num
            for i in range(total_count):
                data, address = self.ConnectionSocket.recvfrom(_udp_clip_num)
                img_bytes += data
                data_count += len(data)
            if left > 0:
                data, address = self.ConnectionSocket.recvfrom(left)
                img_bytes += data
                data_count += len(data)
        try:
            if self.bWith_Flag:
                recv_timestamp = self.recv_timestamp

            img = np.asarray(bytearray(img_bytes))
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # BGR
            self.image_shape = img.shape
            # t = time.time() * 1000
            # print('end:', t - recv_timestamp)
            return recv_timestamp, img

        except AttributeError:
            self.misread_count += 1
            return None, None

    def __next__(self):
        timestamp, img_path, img_0, img = super(AddressImageLoader, self).__next__()

        b_has_data = True
        while b_has_data:
            b_has_data = self.flush_listen()

        return timestamp, img_path, img_0, img

import struct
import time

import numpy as np
from enum import Enum, unique
from socket import *

from ._masterclass import BaseCameraTransLoader

FLAG_HEAD = 'C///'
_udp_clip_num = 64000

_bytes_num_flag_head = len(FLAG_HEAD)
_bytes_num_timestamp = len(struct.pack('q', 0))
_bytes_num_data_count = len(struct.pack('I', 0))
_total_head_num = _bytes_num_flag_head + _bytes_num_timestamp + _bytes_num_data_count


@unique
class E_SupportConnectionType(Enum):
    TCP = 1
    UDP = 2


class AddressTransLoader(BaseCameraTransLoader):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AddressTransLoader, self).__init__(*args, **kwargs)

        address_str_list = self.source.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        assert E_SupportConnectionType[self.connect_type], \
            f'None support connection type {self.connect_type}. Please check it.'
        self.address = (ip, port)
        self.len = 1

        self.ConnectionSocket = None
        self.b_socket_alive = True

        self.recv_timestamp = 0
        self.data_count_total = 0

        self.misread_count = 0

    def pre_process(self) -> bool:
        super(AddressTransLoader, self).pre_process()
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

    def connect(self):
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
        super(AddressTransLoader, self).read_action(idx)
        self.ConnectionSocket.settimeout(1)

        trans = None

        while self.b_socket_alive:
            try:
                if self.bWith_Flag:
                    if not self.recv_flag():
                        continue
                self.recv_timestamp, trans = self.recv_data()
                if isinstance(trans, np.ndarray):
                    return self.recv_timestamp, self.source, trans
                else:
                    pass

            except timeout:
                self.b_socket_alive = False

            except OSError:
                pass

        return self.recv_timestamp, self.source, trans

    def recv_flag(self) -> bool:
        flag_data = self.ConnectionSocket.recv(_udp_clip_num)
        if flag_data:
            if len(flag_data) != _total_head_num:
                return False
            flag_head, self.recv_timestamp, self.data_count_total = \
                struct.unpack(f'={_bytes_num_flag_head}sqI', flag_data)
            flag_head = flag_head.decode("utf-8", "ignore")
            return flag_head and flag_head == FLAG_HEAD
        else:
            return False

    def recv_data(self) -> (int, list):
        trans = []
        data_count = 0
        trans_bytes = b''
        recv_timestamp = int(time.time() * 1000)
        total_count_tmp = (self.data_count_total+8) if self.bWith_Flag else self.data_count_total

        if self.connect_type == E_SupportConnectionType.TCP.name:
            while data_count < total_count_tmp:
                data = self.ConnectionSocket.recv(total_count_tmp)
                trans_bytes += data
                data_count += len(data)

        elif self.connect_type == E_SupportConnectionType.UDP.name:
            total_count = total_count_tmp // _udp_clip_num
            left = total_count_tmp % _udp_clip_num
            for i in range(total_count):
                data, address = self.ConnectionSocket.recvfrom(_udp_clip_num)
                trans_bytes += data
                data_count += len(data)
            if left > 0:
                data, address = self.ConnectionSocket.recvfrom(left)
                trans_bytes += data
                data_count += len(data)
        try:
            if not self.bWith_Flag:
                time_bytes_array = trans_bytes[:8]
                recv_timestamp = struct.unpack('=Q', bytearray(time_bytes_array))
                trans_bytes = trans_bytes[8:]

            for i in range(4 * 4):
                each_data = trans_bytes[i * 4: (i + 1) * 4]
                each_data = struct.unpack('f', each_data)[0]
                trans.append(each_data)
            trans = np.asarray(trans)
            trans = trans.reshape((4, 4))
            return recv_timestamp, trans

        except AttributeError:
            self.misread_count += 1
            return recv_timestamp, None

    def flush_listen(self) -> bool:
        self.ConnectionSocket.settimeout(0)
        while True:
            try:
                data = self.ConnectionSocket.recv(51200)
                if len(data) < 51200:
                    return False
            finally:
                return False

    def __next__(self):
        super(AddressTransLoader, self).__next__()
        if not self.b_socket_alive:
            raise StopIteration
        return self.read_action(self.count)

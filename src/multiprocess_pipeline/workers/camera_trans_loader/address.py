import traceback
import struct
import time

import numpy as np
from enum import Enum, unique
from socket import *

from ._masterclass import BaseCameraTransLoader

FLAG = 'C///'

_bytes_num_flag = len(FLAG)
_bytes_num_timestamp = len(struct.pack('q', 0))
_bytes_num_data_count = len(struct.pack('I', 0))
_total_head_num = _bytes_num_flag + _bytes_num_timestamp + _bytes_num_data_count
_udp_clip_num = 64000


@unique
class ESupportConnectionType(Enum):
    TCP = 1
    UDP = 2


class AddressTransLoader(BaseCameraTransLoader):
    def __init__(self,
                 *args,
                 **kwargs):
        super(AddressTransLoader, self).__init__(*args, **kwargs)

        address_str_list = self.source.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        assert ESupportConnectionType[self.connect_type], \
            f'None support connection type {self.connect_type}. Please check it.'
        self.address = (ip, port)
        self.len = 1

        self.ConnectionSocket = None
        self.b_socket_alive = True

        self.connect()

        self.timestamp = []
        self.trans = []

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

    def read_trans(self, idx) -> (int, str, np.ndarray):
        super(AddressTransLoader, self).read_trans(idx)
        trans_path = self.source
        trans = []
        timestamp = time.time()

        if self.connect_type == ESupportConnectionType.UDP.name:
            self.ConnectionSocket.setblocking(False)
            self.ConnectionSocket.settimeout(1)
            try:
                clip_num = _udp_clip_num
                flag_data, address = self.ConnectionSocket.recvfrom(_total_head_num)
                if flag_data and FLAG in flag_data.decode("utf-8", "ignore"):
                    flag = flag_data
                    timestamp = int(flag[4:8])
                    data_count_total = int(flag[-_bytes_num_data_count:])
                    data_count = 0
                    trans_bytes = b''

                    if data_count_total > 0:
                        total_count = data_count_total // clip_num
                        left = data_count_total % clip_num
                        for i in range(total_count):
                            data, address = self.ConnectionSocket.recvfrom(clip_num)
                            trans_bytes += data
                            data_count += len(data)
                        if left > 0:
                            data, address = self.ConnectionSocket.recvfrom(left)
                            trans_bytes += data
                            data_count += len(data)

                        try:
                            for i in range(4*4):
                                each_data = trans_bytes[i*4: (i+1)*4]
                                each_data = struct.unpack('f', each_data)[0]
                                trans.append(each_data)
                        except:
                            pass
            except timeout or OSError:
                self.b_socket_alive = False

        elif self.connect_type == ESupportConnectionType.TCP.name:
            try:
                data = self.ConnectionSocket.recv(_bytes_num_flag)
                if b'\x00\x00' in data:
                    pass
                if data and data.decode("utf-8", "ignore") == FLAG:
                    flag_data = self.ConnectionSocket.recv(_total_head_num - _bytes_num_flag)
                    flag = flag_data.decode("utf-8", "ignore")
                    timestamp = int(flag[0:_bytes_num_timestamp])
                    data_count_total = int(flag[-_bytes_num_data_count:])
                    data_count = 0
                    trans_bytes = b''

                    if data_count_total > 0:
                        while data_count < data_count_total:
                            data = self.ConnectionSocket.recv(data_count_total)
                            trans_bytes += data
                            data_count += len(data)

                        for i in range(4 * 4):
                            each_data = trans_bytes[i * 4: (i + 1) * 4]
                            each_data = struct.unpack('f', each_data)[0]
                            trans.append(each_data)
                elif not data:
                    self.b_socket_alive = False
            except Exception:
                traceback.print_exc()
                self.b_socket_alive = False

        assert trans is not None, f'Failed to load trans @{trans_path}'
        trans = np.asarray(trans)
        trans = trans.reshape((4, 4))
        return timestamp, trans_path, trans

    def __next__(self):
        super(AddressTransLoader, self).__next__()
        if not self.b_socket_alive:
            raise StopIteration
        return self.read_trans(self.count)

import struct
import numpy as np
from enum import Enum, unique
from socket import *

from . import BaseCameraTransLoader

FLAG_HEAD = 'C///'
_udp_clip_num = 64000

_bytes_num_flag_head = len(FLAG_HEAD)
_bytes_num_timestamp = len(struct.pack('q', 0))
_bytes_num_data_count = len(struct.pack('I', 0))

_total_head_num = _bytes_num_flag_head + \
                  _bytes_num_timestamp + \
                  _bytes_num_data_count


@unique
class E_SupportConnectionType(Enum):
    TCP = 1
    UDP = 2


class AddressTransLoader_All(BaseCameraTransLoader):
    def __init__(self,
                 *args,
                 name_list: list = None,
                 **kwargs):
        super(AddressTransLoader_All, self).__init__(*args, **kwargs)

        self.name_list = name_list
        self.camera_num = len(name_list)

        address_str_list = self.source.split(':')
        self.connect_type, ip, port = address_str_list[0], address_str_list[1], int(address_str_list[2])
        assert E_SupportConnectionType[self.connect_type], \
            f'None support connection type {self.connect_type}. Please check it.'
        self.address = (ip, port)
        self.len = 1

        self.ConnectionSocket = None
        self.b_socket_alive = True

        self.connect()

        self.timestamp = []
        self.trans = []

    def connect(self):
        if self.connect_type == E_SupportConnectionType.TCP.name:
            server_socket = socket(AF_INET, SOCK_STREAM)
            server_socket.bind(self.address)
            server_socket.listen()

            self.ConnectionSocket, address = server_socket.accept()
        elif self.connect_type == E_SupportConnectionType.UDP.name:
            server_socket = socket(AF_INET, SOCK_DGRAM)
            server_socket.bind(self.address)
            self.ConnectionSocket = server_socket

            first_data = None
            while not first_data:
                self.b_socket_alive = True
                first_data = self.read_trans(0)[0]


        else:
            raise

        self.ConnectionSocket.settimeout(60)

    def read_trans(self, idx) -> (int, str, np.ndarray):
        super(AddressTransLoader_All, self).read_trans(idx)

        trans = {}
        timestamp = {}

        self.ConnectionSocket.settimeout(1)

        while self.b_socket_alive:
            try:
                flag_data = self.ConnectionSocket.recv(_total_head_num)
                if flag_data:
                    flag_head, recv_timestamp, data_count_total = struct.unpack(f'={_bytes_num_flag_head}sqI', flag_data)
                    flag_head = flag_head.decode("utf-8", "ignore")
                    if flag_head == FLAG_HEAD:
                        data_count = 0
                        trans_bytes = b''

                        if self.connect_type == E_SupportConnectionType.TCP.name:
                            while data_count < data_count_total:
                                data = self.ConnectionSocket.recv(data_count_total)
                                trans_bytes += data
                                data_count += len(data)
                        elif self.connect_type == E_SupportConnectionType.UDP.name:
                            total_count = data_count_total // _udp_clip_num
                            left = data_count_total % _udp_clip_num
                            for i in range(total_count):
                                data = self.ConnectionSocket.recv(_udp_clip_num)
                                trans_bytes += data
                                data_count += len(data)
                            if left > 0:
                                data = self.ConnectionSocket.recv(left)
                                trans_bytes += data
                                data_count += len(data)

                        for i_cam in range(self.camera_num):
                            each_data = trans_bytes[i_cam * 16 * 4: (i_cam + 1) * 16 * 4]
                            each_data = struct.unpack('=16f', each_data)
                            each_data = np.asarray(list(each_data))
                            trans[self.name_list[i_cam]] = each_data.reshape((4, 4))
                            timestamp[self.name_list[i_cam]] = recv_timestamp

                        return timestamp, self.source, trans

                    elif not flag_head:
                        self.b_socket_alive = False
                else:
                    self.b_socket_alive = False
            except timeout:
                self.b_socket_alive = False
            except OSError:
                pass

        return timestamp, self.source, trans

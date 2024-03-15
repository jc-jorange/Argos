import os
import time
import datetime
from multiprocessing import Process
import torch

from lib.utils.logger import ALL_LoggerContainer
from socket import *
import struct


class PostProcess(Process):
    class Data_strcuture:
        x1, y1, w, h = -1, -1, -1, -1
        xc, yc = -1, -1

    def __init__(self,
                 mode: str,
                 path: str,
                 sub_process_dict: dict,
                 pipe_dict: dict,
                 ):
        super().__init__()
        self.path = path
        self.mode = mode
        self.sub_process_dict = sub_process_dict
        self.pipe_dict = pipe_dict
        self.keep_process = True

        self.result_dict = {}

        if self.mode != "None":
            HOST = path.split(':')[0]
            PORT = int(path.split(':')[1])
            ADDRESS = (HOST, PORT)
            self.address = ADDRESS

            if self.mode == "Outer_TCP":
                ServerSocket = socket(AF_INET, SOCK_STREAM)
                ServerSocket.bind(ADDRESS)
                ServerSocket.listen()
                ALL_LoggerContainer.logger_dict[os.getpid()].info("Waiting for TCP connection at {}:{}".format(HOST, PORT))

                self.ConnectionSocket, address = ServerSocket.accept()
                ALL_LoggerContainer.logger_dict[os.getpid()].info(
                    "Successfully connected to {} and connection on {}".format(ADDRESS, address))
            elif self.mode == "Outer_UDP":
                ServerSocket = socket(AF_INET, SOCK_DGRAM)
                self.ConnectionSocket = ServerSocket
                ALL_LoggerContainer.logger_dict[os.getpid()].info("UDP connection at {}:{}".format(HOST, PORT))
            else:
                raise


    def has_subprocess_alive(self):
        re = False
        for k, v in self.sub_process_dict.items():
            if v.is_alive():
                re = True
            else:
                pass
        return re

    def process_content(self):
        for k, v in self.pipe_dict.items():
            if v.poll():
                results_dict_get = v.recv()
                class_dict = {}
                for cls_id in range(len(results_dict_get)):  # process each object class
                    cls_results = results_dict_get[cls_id]
                    id_dict = {}
                    for frame_id, tlwhs, track_ids, scores in cls_results:
                        for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                            if track_id < 0:
                                continue
                            data = self.Data_strcuture()
                            data.x1, data.y1, data.w, data.h = tlwh
                            data.xc, data.yc = data.x1 + data.w / 2, data.y1 + data.h / 2

                            id_dict[track_id] = data
                    class_dict[cls_id] = id_dict
                self.result_dict[k] = class_dict

        target_class = 0
        content = (-1.0, -1.0)
        frequency = 120.0
        try:
            id_result = list(self.result_dict.values())[0][target_class]
            track_result = list(id_result.values())[0]
            content = (track_result.xc, track_result.yc)
        except:
            pass

        # time.sleep(1 / frequency)

        msg = struct.pack('@ff', content[0], content[1])
        if self.mode == "Outer_TCP":
            self.ConnectionSocket.send(msg)
        elif self.mode == "Outer_UDP":
            self.ConnectionSocket.sendto(msg, self.address)

    def process(self):
        while self.has_subprocess_alive():
            self.process_content()

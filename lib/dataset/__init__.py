import glob
import math
import os
import os.path as osp
import random
import copy
import time
import warnings
import traceback

import cv2
import numpy as np
from socket import *
import multiprocessing

from collections import OrderedDict, defaultdict
from lib.utils.logger import ALL_LoggerContainer
from lib.utils.image import gaussian_radius, draw_umich_gaussian
from lib.utils.utils import xyxy2xywh
from lib.model.model_config import E_arch_position, E_model_part_input_info
from lib.dataset.utils.utils import create_gamma_img, clear_socket_buffer
import lib.multiprocess.Shared as Sh
from  lib.multiprocess.MP_ImageReceiver import EImageInfo

def letterbox(img,
              height=608,
              width=1088,
              color=(127.5, 127.5, 127.5)):
    """
    resize a rectangular image to a padded rectangular
    :param img:
    :param height:
    :param width:
    :param color:
    :return:
    """
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])

    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) * 0.5  # width padding
    dh = (height - new_shape[1]) * 0.5  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    # padded rectangular
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


# for inference
class LoadData:
    def __init__(self, idx, data_type, path, shared_image_dict: Sh.SharedDict, img_size=(1088, 608)):
        """
        Load image data.

        :param str data_type: Image or Video or Stream
        :param str path: Data path or Stream address
        :param tuple img_size: Target image size
        """
        self.frame_rate = 10  # no actual meaning here
        self.data_type = data_type
        self.path = path
        self.logger = ALL_LoggerContainer.logger_dict[os.getpid()]
        self.last_process_time = -1
        self.idx = idx
        # self.pipe_ImageReceiver_out = pipe
        self.shared_image_dict = shared_image_dict

        if self.data_type == 'Image':
            if type(path) == str:
                if os.path.isdir(path):
                    image_format = ['.jpg', '.jpeg', '.png', '.tif', '.exr']
                    self.files = sorted(glob.glob('%s/*.*' % path))
                    self.files = list(filter(lambda x: os.path.splitext(x)[
                                                           1].lower() in image_format, self.files))
                elif os.path.isfile(path):
                    self.files = [path]
            elif type(path) == list:
                self.files = path
            self.nF = len(self.files)  # number of image files
            assert self.nF > 0, 'No images found in ' + path

        elif self.data_type == 'Video':
            cap = cv2.VideoCapture(path)

            self.frame_rate = int(round(cap.get(cv2.CAP_PROP_FPS)))
            self.vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.nF = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.logger.info('Length of the video: {:d} frames'.format(self.nF))

        elif self.data_type == 'Address':
            self.nF = 1

        self.shm = None

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        if self.data_type == 'Image' or self.data_type == 'Video':
            if self.count == len(self):
                raise StopIteration
        elif self.data_type == 'Address':
            try:
                self.shared_image_dict.read(EImageInfo.Data)
            except KeyError:
                raise StopIteration

        return self.read_image(self.count)

    def __getitem__(self, idx):
        idx = idx % len(self)

        return self.read_image(idx)

    def __len__(self):
        return self.nF

    def read_image(self, idx):
        img_0 = None
        img_path = ''
        start_time = time.perf_counter()
        if self.data_type == 'Image':
            img_path = self.files[idx]
            img_0 = cv2.imread(img_path)  # BGR
            assert img_0 is not None, 'Failed to load ' + img_path

        elif self.data_type == 'Video':
            img_path = str(idx)
            cap = cv2.VideoCapture(self.path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            res, img_0 = cap.read()  # BGR
            assert img_0 is not None, 'Failed to load frame ' + img_path

        elif self.data_type == 'Address':
            img_path = 'Current frame'
            self.shm, img_ad = Sh.read_from_shm(Sh.NAME_shm_img + str(self.idx), (200,320,3), np.uint8)
            img_0 = np.ascontiguousarray(np.copy(img_ad))

        end_time = time.perf_counter()
        only_read = end_time - start_time
        # print('only read time: %f' % (only_read))

        # Padded resize
        img, _, _, _ = letterbox(img_0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        end_time = time.perf_counter()
        read_all = end_time - start_time
        # print('read all process time: %f' % (read_all))

        process_time = time.perf_counter()
        if self.last_process_time <0:
            self.last_process_time = process_time
        else:
            read_gap = (process_time - self.last_process_time)
            self.last_process_time = process_time
            # print("read gap: %f" % (read_gap) )

        if self.shm:
            self.shm.close()
        return img_path, img, img_0


def random_affine(img, targets=None,
                  degrees=(-10, 10),
                  translate=(.1, .1),
                  scale=(.9, 1.1),
                  shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
              img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
              img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        if len(targets) > 0:
            n = targets.shape[0]
            points = targets[:, 2:6].copy()
            area0 = (points[:, 2] - points[:, 0]) * \
                    (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate(
                (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)),
                            abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            # np.clip(xy[:, 0], 0, width, out=xy[:, 0])
            # np.clip(xy[:, 2], 0, width, out=xy[:, 2])
            # np.clip(xy[:, 1], 0, height, out=xy[:, 1])
            # np.clip(xy[:, 3], 0, height, out=xy[:, 3])
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            targets = targets[i]
            targets[:, 2:6] = xy[i]
            targets = targets[targets[:, 2] < width]
            targets = targets[targets[:, 4] > 0]
            targets = targets[targets[:, 3] < height]
            targets = targets[targets[:, 5] > 0]

        return imw, targets, M
    else:
        return imw


# for training
class LoadImagesAndLabels:  # for training
    def __init__(self,
                 path,
                 img_size=(1088, 608),
                 augment=False,
                 transforms=None):
        """
        :param path:
        :param img_size:
        :param augment:
        :param transforms:
        """
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids')
                                .replace('.png', '.txt')
                                .replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files

        self.width = img_size[0]
        self.height = img_size[1]

        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path, width=None, height=None):
        """
        Image data convert and enhance; Label format
        :param img_path:
        :param label_path:
        :param height:
        :param width:
        :return:
        """
        # Input resolution
        if height is None or width is None:
            height = self.height
            width = self.width

        # Read image data to numpy array, in BGR order
        img = cv2.imread(img_path)  # cv(numpy): BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))

        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, pad_w, pad_h = letterbox(img, height=height, width=width)  # resizing and padding

        # Load labels
        if os.path.isfile(label_path):
            with warnings.catch_warnings():  # No warnings for empty label file(txt)
                warnings.simplefilter("ignore")
                labels_0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

                # reformat xywh to pixel xyxy(x1, y1, x2, y2) format
                labels = labels_0.copy()  # deep copy
                labels[:, 2] = ratio * w * (labels_0[:, 2] - labels_0[:, 4] / 2) + pad_w  # x1
                labels[:, 3] = ratio * h * (labels_0[:, 3] - labels_0[:, 5] / 2) + pad_h  # y1
                labels[:, 4] = ratio * w * (labels_0[:, 2] + labels_0[:, 4] / 2) + pad_w  # x2
                labels[:, 5] = ratio * h * (labels_0[:, 3] + labels_0[:, 5] / 2) + pad_h  # y2
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(img, labels,
                                           degrees=(-5, 5),
                                           translate=(0.10, 0.10),
                                           scale=(0.50, 1.20))

        plot_flag = False
        if plot_flag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T,
                     labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        num_labels = len(labels)
        if num_labels > 0:
            # convert xyxy to xywh(center_x, center_y, b_w, b_h)
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())

            # normalize to 0~1
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if num_labels > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


# ---------- Predefined multi-scale input image width and height list
Input_WHs = [
    [640, 342],   # 0
    [672, 352],   # 1
    [704, 384],   # 2
    [736, 416],   # 3
    [768, 448],   # 4
    [800, 480],   # 5
    [832, 512],   # 6
    [864, 544],   # 7
    [896, 576],   # 8
    [928, 608],   # 9
    [960, 640],   # 10
    [992, 672],   # 11
    [1064, 704],  # 12
    [1064, 736],  # 13
    [1064, 608],  # 14
    [1088, 608]   # 15
]  # total 16 scales with floating aspect ratios


class TrainingDataset(LoadImagesAndLabels):
    """
    multi scale for training
    """
    mean = None
    std = None
    gs = 32

    def __init__(self,
                 opt,
                 info_data,
                 root,
                 paths,
                 augment=False,
                 transforms=None):
        """
        :param opt: Parser for all options
        :param info_data: information data for model
        :param dict root: Dataset root
        :param dict paths: Training part dir in dataset root
        :param augment:
        :param transforms: Image data transformations, default is transforms.ToTensor
        """
        self.logger = ALL_LoggerContainer.logger_dict[multiprocessing.current_process().name]
        self.opt = opt
        self.info_data = info_data
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = info_data.classes_max_num

        # # make sure img_size equal to opt.input_wh
        # if opt.input_wh[0] != img_size[0] or opt.input_wh[1] != img_size[1]:
        #     opt.input_wh[0], opt.input_wh[1] = img_size[0], img_size[1]
        #
        # # default input width and height
        # self.default_input_wh = opt.input_wh

        # define mapping from batch idx to scale idx
        self.batch_i_to_scale_i = defaultdict(int)

        # generate img and label file path lists
        self.paths = paths
        for ds, path in self.paths.items():  # every sub dataset ds: sub dataset name, path: dataset dir
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [x.replace('images', 'labels_with_ids') # need to do: custom label dir
                                        .replace('.png', '.txt')
                                        .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

            self.logger.info('Total {} image files in {} dataset.'.format(len(self.label_files[ds]), ds))

            # set train image  floor with min size unit gs
            if opt.gen_multi_scale:
                img = cv2.imread(self.img_files[ds][0])
                self.width = int((img.shape[1])//self.gs)*self.gs
                self.height = int((img.shape[0])//self.gs)*self.gs
            else:
                self.width = 1088
                self.height = 608

        for ds, label_paths in self.label_files.items():  # every sub dataset
            max_ids_dict = defaultdict(int)  # cls_id => max track id

            # label file in every sub dataset
            for lp in label_paths:
                if not os.path.isfile(lp):
                    self.logger.warning('invalid label file {}.'.format(lp))
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    lb = np.loadtxt(lp)
                    if len(lb) < 1:  # empty label file
                        continue

                    lb = lb.reshape(-1, 6)
                    for item in lb:  # every item in label
                        if item[1] > max_ids_dict[int(item[0])]:  # item[0]: cls_id, item[1]: track id
                            max_ids_dict[int(item[0])] = item[1]

            # track id number
            self.tid_num[ds] = {cls: max_ids_dict[cls] + 1 for cls in max_ids_dict.keys()}  # dict made by cls_id

        self.tid_start_idx_of_cls_ids = defaultdict(dict)
        last_idx_dict = defaultdict(int)
        for ds, v in self.tid_num.items():  # for every sub dataset, ds: dataset name, v: max_ids_dict
            for cls_id, id_num in v.items():  # every class, v is max_ids_dict
                self.tid_start_idx_of_cls_ids[ds][cls_id] = last_idx_dict[cls_id]
                last_idx_dict[cls_id] += int(id_num)

        self.nID_dict = defaultdict(int)
        for cls_id, id_num in last_idx_dict.items():
            self.nID_dict[cls_id] = int(last_idx_dict[cls_id]+1)  # track ids number for each class
            self.logger.info('Total {:d} IDs of class {}'.format(self.nID_dict[cls_id], cls_id))

        self.nds = [len(x) for x in self.img_files.values()]  # every sub dataset image files number
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)  # number fot all image in training
        self.max_objs = info_data.objects_max_num  # max target number in each image
        self.augment = augment
        self.transforms = transforms

        self.input_multi_scales = None

        if opt.gen_multi_scale:  # whether to generate multi-scales while keeping aspect ratio
            self.gen_multi_scale_input_whs()

            # rand scale the first time
            self.rand_scale()
            self.logger.info('Total {:d} multi-scales:'.format(len(self.input_multi_scales)))

    def rand_scale(self):
        # randomly generate mapping from batch idx to scale idx
        self.num_batches = self.nF // self.opt.batch_size + 1
        for batch_i in range(self.num_batches):
            rand_batch_idx = np.random.randint(0, self.num_batches)
            scale = len(Input_WHs) if self.input_multi_scales is None else len(self.input_multi_scales)
            rand_scale_idx = rand_batch_idx % scale
            self.batch_i_to_scale_i[batch_i] = rand_scale_idx

    def gen_multi_scale_input_whs(self, num_scales=256, min_ratio=0.67, max_ratio=1.1):
        """
        generate input multi scale image sizes(w, h), keep default aspect ratio
        :param num_scales:
        :return:
        """
        gs = 32  # grid size

        self.input_multi_scales = [x for x in Input_WHs if not (x[0] % gs or x[1] % gs)]
        self.input_multi_scales.append([self.width, self.height])

        # ----- min scale and max scale
        # keep default aspect ratio
        self.default_aspect_ratio = self.height / self.width

        # min scale
        min_width = math.ceil(self.width * min_ratio / gs) * gs
        min_height = math.ceil(self.height * min_ratio / gs) * gs
        self.input_multi_scales.append([min_width, min_height])

        # max scale
        max_width = math.ceil(self.width * max_ratio / gs) * gs
        max_height = math.ceil(self.height * max_ratio / gs) * gs
        self.input_multi_scales.append([max_width, max_height])

        # other scales
        # widths = list(range(min_width, max_width + 1, int((max_width - min_width) / num_scales)))
        # heights = list(range(min_height, max_height + 1, int((max_height - min_height) / num_scales)))
        widths = list(range(min_width, max_width + 1, 1))
        heights = list(range(min_height, max_height + 1, 1))
        widths = [width for width in widths if not (width % gs)]
        heights = [height for height in heights if not (height % gs)]
        if len(widths) < len(heights):
            for width in widths:
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        elif len(widths) > len(heights):
            for height in heights:
                width = math.ceil(height / self.default_aspect_ratio / gs) * gs
                if [width, height] in self.input_multi_scales:
                    continue
                self.input_multi_scales.append([width, height])
        else:
            for width, height in zip(widths, heights):
                if [width, height] in self.input_multi_scales:
                    continue
                height = math.ceil(width * self.default_aspect_ratio / gs) * gs
                self.input_multi_scales.append([width, height])

        if len(self.input_multi_scales) < 2:
            self.input_multi_scales = None
            self.logger.warning('generate multi-scales failed(keeping aspect ratio)')
        else:
            self.input_multi_scales.sort(key=lambda x: x[0])

    def shuffle(self):
        """
        random shuffle the dataset
        :return:
        """
        tmp_img_files = copy.deepcopy(self.img_files)
        for ds, path in self.paths.items():
            ds_n_f = len(self.img_files[ds])  # number of files of this sub-dataset
            orig_img_files = self.img_files[ds]

            # re-generate ids
            uesd_ids = []
            for i in range(ds_n_f):
                new_idx = np.random.randint(0, ds_n_f)
                if new_idx in uesd_ids:
                    continue

                uesd_ids.append(new_idx)
                tmp_img_files[ds][i] = orig_img_files[new_idx]

        self.img_files = tmp_img_files  # re-evaluate img_files
        for ds, path in self.paths.items():  # re-evaluate corresponding label files
            self.label_files[ds] = [x.replace('images', 'labels_with_ids')
                                        .replace('.png', '.txt')
                                        .replace('.jpg', '.txt')
                                    for x in self.img_files[ds]]

    def __getitem__(self, idx):
        batch_i = idx // int(self.opt.batch_size)
        scale_idx = self.batch_i_to_scale_i[batch_i]
        if self.input_multi_scales is None:
            width, height = self.width, self.height
        else:
            width, height = self.input_multi_scales[scale_idx]

        # start index for every sub dataset
        for i, c in enumerate(self.cds):
            if idx >= c:
                ds = list(self.label_files.keys())[i]  # dataset at idx
                start_index = c

        img_path = self.img_files[ds][idx - start_index]
        label_path = self.label_files[ds][idx - start_index]

        # Get image data and label: using multi-scale input image size
        imgs, labels, img_path, (input_h, input_w) = self.get_data(img_path, label_path, width, height)

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                cls_id = int(labels[i][0])
                labels[i, 1] += self.tid_start_idx_of_cls_ids[ds][cls_id]

        output_h = imgs.shape[1] // self.info_data.input_info[E_arch_position(0).name][E_model_part_input_info(1)][-1]
        output_w = imgs.shape[2] // self.info_data.input_info[E_arch_position(0).name][E_model_part_input_info(1)][-1]

        # actual target number in image
        num_objs = labels.shape[0]

        # --- GT of detection
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)  # C×H×W
        wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs,), dtype=np.int64)  # K objects
        # only calculate reg loss at pixel with target in feature map
        reg_mask = np.zeros((self.max_objs,), dtype=np.uint8)

        ids = np.zeros((self.max_objs,), dtype=np.int64)
        cls_tr_ids = np.zeros((self.num_classes, output_h, output_w), dtype=np.int64)
        cls_id_map = np.full((1, output_h, output_w), -1, dtype=np.int64)  # 1×H×W

        # Gauss function definition
        # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        draw_gaussian = draw_umich_gaussian

        # iter every ground truth target
        for k in range(min(num_objs, self.max_objs)):  # actual target number in image
            label = labels[k]

            # bbox output GT
            #                       0        1        2       3
            bbox = label[2:]  # center_x, center_y, bbox_w, bbox_h

            # class from 0
            cls_id = int(label[0])

            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)

            w, h = bbox[2], bbox[3]

            if h > 0 and w > 0:
                # heat-map radius
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))  # radius >= 0
                # radius = self.opt.hm_gauss if self.opt.mse_loss else radius

                # bbox center coordinate
                ct = np.array([bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)  # floor int

                # draw gauss weight for heat-map
                draw_gaussian(hm[cls_id], ct_int, radius)  # hm

                # --- GT of detection
                wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                        bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]

                # target index in feature map
                ind[k] = ct_int[1] * output_w + ct_int[0]  # feature map index:y*w+x

                # offset regression
                reg[k] = ct - ct_int
                reg_mask[k] = 1

                # --- GT of ReID
                cls_id_map[0][ct_int[1], ct_int[0]] = cls_id  # 1×H×W

                cls_tr_ids[cls_id][ct_int[1]][ct_int[0]] = label[1]

                ids[k] = label[1]

        ret = {'input': imgs,
               'hm': hm,
               'reg': reg,
               'wh': wh,
               'ind': ind,
               'reg_mask': reg_mask,
               'ids': ids,
               'cls_id_map': cls_id_map,  # id at every (x,y) in feature map
               'cls_tr_ids': cls_tr_ids,
               'meta': label_path
               }
        return ret

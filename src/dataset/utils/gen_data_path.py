import os
import glob
import math

# def gen_data_path(root_path):
#     mot_path = 'MOT17/train'
#     real_path = os.path.join(root_path, mot_path)
#     seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
#     with open('/home/yfzhang/PycharmProjects/fairmot/src/data/mot17.half', 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             seq_path = os.path.join(seq_path, 'img1')
#             images = sorted(glob.glob(seq_path + '/*.jpg'))
#             len_all = len(images)
#             len_half = int(len_all / 2)
#             for i in range(len_half):
#                 image = images[i]
#                 print(image[22:], file=f)
#     f.close()

def gen_data_path(root_path,
                  mot_path,
                  dataset_name,
                  train_ratio=1.0,
                  seq_folder_name='',
                  endwith='',
                  image_formate='png'):
    real_path = os.path.join(root_path, mot_path)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith(endwith)]

    data_dir = '/home/jc/PythonScripts/Argus/src/dataset/data_path'
    trainfile = dataset_name + '.trainall'
    halffile = dataset_name + '.half'
    valfile = dataset_name + '.val'

    if train_ratio > 1.0 or train_ratio < 0.0:
        raise ValueError('Ratio not in 0~1.0 range')
    else:
        with open(os.path.join(data_dir, trainfile), 'w') as train,\
                open(os.path.join(data_dir, halffile), 'w') as half, \
                open(os.path.join(data_dir, valfile), 'w') as val:
            for seq_name in seq_names:
                seq_path = os.path.join(real_path, seq_name, seq_folder_name)
                images = sorted(glob.glob(seq_path + '/*.' + image_formate))
                len_all = len(images)
                for count, image in enumerate(images):
                    print(image[len(root_path)+1:], file=train)
                    if count < math.floor(len_all * train_ratio):
                        print(image[len(root_path) + 1:], file=half)
                    else:
                        print(image[len(root_path) + 1:], file=val)
        train.close(), half.close(), val.close()

# def gen_data_path_custom(root_path, type):
#     mot_path = 'Custom/Train/images/Standard00'
#     real_path = os.path.join(root_path, mot_path)
#     seq_names = [s for s in sorted(os.listdir(real_path))]
#
#     data_dir = './data'
#     filename = 'custom_FHD.' + type
#     with open(os.path.join(data_dir, filename), 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             images = sorted(glob.glob(seq_path + '/*.png'))
#             for image in images:
#                 print(image[len(root_path)+1:], file=f)
#     f.close()
#
#
# def gen_data_path_mot17(root_path, type):
#     mot_path = 'MOT17/images/train'
#     real_path = os.path.join(root_path, mot_path)
#     seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith('SDP')]
#
#     data_dir = './data'
#     filename = 'mot17.' + type
#     with open(os.path.join(data_dir, filename), 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             seq_path = os.path.join(seq_path, 'img1')
#             images = sorted(glob.glob(seq_path + '/*.jpg'))
#             for image in images:
#                 print(image[len(root_path)+1:], file=f)
#     f.close()
#
# def gen_data_path_mot20(root_path, type):
#     mot_path = 'MOT20/images/train'
#     real_path = os.path.join(root_path, mot_path)
#     seq_names = [s for s in sorted(os.listdir(real_path))]
#
#     data_dir = './data'
#     filename = 'mot20.' + type
#     with open(os.path.join(data_dir, filename), 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             seq_path = os.path.join(seq_path, 'img1')
#             images = sorted(glob.glob(seq_path + '/*.jpg'))
#             for image in images:
#                 print(image[len(root_path)+1:], file=f)
#     f.close()
#
# def gen_data_path_VisDrone2019(root_path, type):
#     visdrone_path = 'VisDrone2019-MOT/images/VisDrone2019-MOT-train'
#     real_path = os.path.join(root_path, visdrone_path, 'sequences')
#     seq_names = [s for s in sorted(os.listdir(real_path))]
#
#     data_dir = './data'
#     filename = 'VisDrone2019.' + type
#     with open(os.path.join(data_dir, filename), 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             images = sorted(glob.glob(seq_path + '/*.jpg'))
#             for image in images:
#                 print(image[len(root_path)+1:], file=f)
#     f.close()
#
# def gen_data_path_KITTI_Tracking2D(root_path, type):
#     KITTI_Tracking2D_path = 'KITTI_Object_Tracking_Evaluation/images/training'
#     real_path = os.path.join(root_path, KITTI_Tracking2D_path, 'image_02')
#     seq_names = [s for s in sorted(os.listdir(real_path))]
#
#     data_dir = './data'
#     filename = 'KITTI_Tracking2D.' + type
#     with open(os.path.join(data_dir, filename), 'w') as f:
#         for seq_name in seq_names:
#             seq_path = os.path.join(real_path, seq_name)
#             images = sorted(glob.glob(seq_path + '/*.png'))
#             for image in images:
#                 print(image[len(root_path)+1:], file=f)
#     f.close()

if __name__ == '__main__':
    root = '/media/jc/Extend/Library/Dataset'
    gen_data_path(root_path=root,
                  mot_path='AutoDataset/Train/images',
                  dataset_name='AutoDataset',
                  seq_folder_name='',
                  train_ratio=1.0,
                  endwith='',
                  image_formate='png')
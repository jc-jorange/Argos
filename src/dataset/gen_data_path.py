import os
import glob
import math
from tqdm import tqdm


def gen_data_path(root_dir: str,
                  mot_path_in_root: str,
                  dataset_name: str,
                  save_dir: str,
                  train_ratio=1.0,
                  seq_folder_name='',
                  seq_endswith='',
                  image_formate='png'):
    real_path = os.path.join(root_dir, mot_path_in_root)
    seq_names = [s for s in sorted(os.listdir(real_path)) if s.endswith(seq_endswith)]

    train_file_name = dataset_name + '.trainall'
    half_file_name = dataset_name + '.half'
    val_file_name = dataset_name + '.val'

    if train_ratio > 1.0 or train_ratio < 0.0:
        raise ValueError('Ratio not in 0~1.0 range')
    else:
        print(f'start generating')
        with open(os.path.join(save_dir, train_file_name), 'w') as train,\
                open(os.path.join(save_dir, half_file_name), 'w') as half, \
                open(os.path.join(save_dir, val_file_name), 'w') as val:
            seq_name: str
            for seq_name in tqdm(seq_names, desc='Total sequences'):
                seq_path = os.path.join(real_path, seq_name, seq_folder_name)
                images = sorted(glob.glob(seq_path + '/*.' + image_formate))
                len_all = len(images)
                if len_all <= 0:
                    raise ValueError(f'No images in dataset path {seq_path}')
                for count, image_path in enumerate(images):
                    path_from_root = os.path.abspath(image_path)[len(os.path.abspath(root_dir)) + 1:]
                    print(path_from_root, file=train)
                    if count < math.floor(len_all * train_ratio):
                        print(path_from_root, file=half)
                    else:
                        print(path_from_root, file=val)
        train.close(), half.close(), val.close()
        print(f'generation over')


if __name__ == '__main__':
    gen_data_path(root_dir='/media/jc/Extend/Library/Dataset/',
                  mot_path_in_root='AutoDataset/RealWorldExperiment/Train/images',
                  dataset_name='CustomDataset',
                  save_dir=r'/home/jc/PythonScripts/Argos/src/dataset/data_path',
                  seq_folder_name='',
                  train_ratio=0.5,
                  seq_endswith='',
                  image_formate='png')

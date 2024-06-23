import os
import glob
import math


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
            for seq_name in seq_names:
                seq_path = os.path.join(real_path, seq_name, seq_folder_name)
                images = sorted(glob.glob(seq_path + '/*.' + image_formate))
                len_all = len(images)
                for count, image in enumerate(images):
                    print(image[len(root_dir) + 1:], file=train)
                    if count < math.floor(len_all * train_ratio):
                        print(image[len(root_dir) + 1:], file=half)
                    else:
                        print(image[len(root_dir) + 1:], file=val)
        train.close(), half.close(), val.close()
        print(f'generation over')


if __name__ == '__main__':
    gen_data_path(root_dir='D:/Library/Dataset',
                  mot_path_in_root='AutoDataset/Experiment_02/Train/images',
                  dataset_name='FuncTest',
                  save_dir=r'/src/dataset/data_path',
                  seq_folder_name='',
                  train_ratio=1.0,
                  seq_endswith='',
                  image_formate='png')

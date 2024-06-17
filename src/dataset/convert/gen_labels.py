import os
import glob
import cv2


def gen_VisDrone2019_label():
    root = '/media/jc/Extend/Library/Dataset/'
    data_dir = 'VisDrone2019-MOT/images/VisDrone2019-MOT-train/sequences'
    real_data_dir = os.path.join(root, data_dir)
    gt_dir = real_data_dir.replace('sequences', 'annotations')
    gt_save_dir = real_data_dir.replace('images', 'labels_with_ids')
    files = os.listdir(gt_dir)
    for i, file in enumerate(files):
        if file.endswith('.txt'):
            sequence_name = file.split('.')[0]
            img = cv2.imread(os.path.join(real_data_dir, sequence_name, '0000001.jpg'))
            sp = img.shape
            i_width = float(sp[1])
            i_height = float(sp[0])

            gt_sequence = os.path.join(gt_save_dir, sequence_name)
            if not os.path.exists(gt_sequence):
                os.makedirs(gt_sequence)

            with open(os.path.join(gt_dir, file), 'r') as f:
                for line in f.readlines():
                    context = line.split(',')
                    frame = context[0].zfill(7)
                    tid = context[1]
                    category = context[7]
                    center_x = (float(context[2]) + float(context[4])/2) / i_width
                    center_y = (float(context[3]) + float(context[5])/2) / i_height
                    w = float(context[4]) / i_width
                    h = float(context[5]) / i_height
                    gt_tmp = [category, tid, str(center_x), str(center_y), str(w), str(h)]
                    gt = ' '.join(gt_tmp) + '\n'

                    with open(os.path.join(gt_sequence + '/%s.txt' % frame), 'a') as gt_f:
                        gt_f.writelines(gt)

    for sequence in os.listdir(real_data_dir):
        for file in os.listdir(os.path.join(real_data_dir, sequence)):
            file_name = file.split('.')[0]
            new_gt = os.path.join(gt_save_dir, sequence)+'/%s.txt'%file_name
            if not os.path.exists(new_gt):
                with open(new_gt, 'w') as f:
                    pass


def gen_KITTI_Tracking2D_label():
    category_map = {
        'DontCare': -1,
        'Car': 0,
        'Van': 1,
        'Truck': 2,
        'Pedestrian': 3,
        'Person': 4,
        'Cyclist': 5,
        'Tram': 6,
        'Misc': 7,
    }
    root = '/media/jc/Extend/Library/Dataset/'
    data_dir = 'KITTI_Object_Tracking_Evaluation/images/training/image_02'
    real_data_dir = os.path.join(root, data_dir)
    gt_dir = real_data_dir.replace('image_02', 'label_02')
    gt_save_dir = real_data_dir.replace('images', 'labels_with_ids')
    files = os.listdir(gt_dir)
    for i, file in enumerate(files):
        if file.endswith('.txt'):
            sequence_name = file.split('.')[0]
            img = cv2.imread(os.path.join(real_data_dir, sequence_name, '000000.png'))
            sp = img.shape
            i_width = float(sp[1])
            i_height = float(sp[0])

            gt_sequence = os.path.join(gt_save_dir, sequence_name)
            if not os.path.exists(gt_sequence):
                os.makedirs(gt_sequence)

            with open(os.path.join(gt_dir, file), 'r') as f:
                for line in f.readlines():
                    context = line.split(' ')
                    if int(context[1]) > 0:
                        frame = context[0].zfill(6)
                        tid = context[1]
                        category = str(category_map[context[2]])
                        left = float(context[6])
                        top = float(context[7])
                        right = float(context[8])
                        bottom = float(context[9])

                        w = right - left
                        h = bottom - top
                        center_x = (left + w/2)
                        center_y = (top + h/2)
                        gt_tmp = [category, tid, str(center_x / i_width), str(center_y / i_height),
                                  str(w / i_width), str(h / i_height)]
                        gt = ' '.join(gt_tmp) + '\n'

                        with open(os.path.join(gt_sequence + '/%s.txt' % frame), 'a') as gt_f:
                            gt_f.writelines(gt)

    for sequence in os.listdir(real_data_dir):
        for file in os.listdir(os.path.join(real_data_dir, sequence)):
            file_name = file.split('.')[0]
            new_gt = os.path.join(gt_save_dir, sequence)+'/%s.txt'%file_name
            if not os.path.exists(new_gt):
                with open(new_gt, 'w') as f:
                    pass

if __name__ == '__main__':
    gen_VisDrone2019_label()

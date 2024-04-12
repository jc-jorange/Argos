import os
import cv2
import numpy as np
from enum import Enum, unique
from collections import defaultdict

# {frame: {subframe: ({class: {ID: ((tlwh), score)}}, fps)}}
S_default_result = {0: {0: ({0: {0: ((0, 0, 0, 0), 0.0)}}, 0.0)}}

@unique
class E_text_result_type(Enum):
    raw = 1
    mot = 2
    kitti = 3


Dict_text_result_name = {
    E_text_result_type.raw: 'result_raw.txt',
    E_text_result_type.mot: 'result_mot.txt',
    E_text_result_type.kitti: 'result_kitti.txt',
}

Dict_text_result_format = {
    E_text_result_type.raw: '{frame},{subframe},{cls_id},{id},{x1},{y1},{x2},{y2},{score},{fps}\n',
    E_text_result_type.mot: '{frame},{id},{x1},{y1},{x2},{y2},{score},{cls_id},1\n',
    E_text_result_type.kitti: '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n',
}

Str_video_result_name = 'video_result.mp4'


def convert_numpy_to_dict(result: np.ndarray, frame: int, subframe: int, fps: float) -> dict:
    result_frame = {}
    result_each_subframe = {}
    result_class = defaultdict(dict)

    valid_position = np.nonzero(result)

    target_num = len(valid_position[0]) // 4
    for i in range(target_num):
        cls = valid_position[0][i * 4]
        target_id = valid_position[1][i * 4]
        x_position = valid_position[2][(i * 4)]
        y_position = valid_position[2][(i * 4) + 1]
        x = result[cls][target_id][x_position]
        y = result[cls][target_id][y_position]
        result_class[cls][target_id] = ((x, y, 0, 0), 1.0)

    result_each_subframe[subframe] = (result_class, fps)
    result_frame[frame] = result_each_subframe

    return result_frame


def get_color(i_class: int, idx: int) -> tuple:
    import random
    seed = i_class * 10000 + idx * 10
    random.seed(seed + 0)
    r = random.randint(0, 255)
    random.seed(seed + 1)
    g = random.randint(0, 255)
    random.seed(seed + 2)
    b = random.randint(0, 255)
    color = (r, g, b)
    return color


def plot_tracks(
        image,
        result_frame: dict,
        frame: int,
):
    """
    :rtype:
    :param image:
    :param result_by_class:
    :param frame:
    :param subframe:
    :param fps:
    :return:
    """

    img = np.ascontiguousarray(np.copy(image))

    text_scale = max(1.0, image.shape[1] / 1200.)  # 1600.
    text_thickness = 1  # 自定义ID文本线宽
    line_thickness = max(2, int(image.shape[1] / 500.))

    subframe_total = list(result_frame.keys())[-1] if len(list(result_frame.keys())) > 0 else 0
    fps = result_frame[list(result_frame.keys())[0]][1] if subframe_total >= 0 else 0.0

    cv2.putText(
        img,
        'frame: %d + %d fps: %.2f' % (frame, subframe_total, fps),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=1
    )

    result_last_subframe = {}
    for each_subframe, result_subframe in result_frame.items():
        result_class = result_subframe[0]
        for i_class, result_by_id in result_class.items():
            for i_id, result in result_by_id.items():
                tlwh, score = result
                x1, y1, w, h = tlwh
                int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))  # x1, y1, x2, y2
                id_text = f'{i_id}'

                color = get_color(i_class, i_id)

                if (w * h) > 0:
                    # draw bbox
                    cv2.rectangle(
                        img=img,
                        pt1=int_box[0:2],  # (x1, y1)
                        pt2=int_box[2:4],  # (x2, y2)
                        color=color,
                        thickness=line_thickness
                                  )
                else:
                    # draw arrow
                    try:
                        last_result = result_last_subframe[i_class][i_id][0]
                        last_point = (int(last_result[0]), int(last_result[1]))
                        cv2.arrowedLine(
                            img=img,
                            pt1=int_box[0:2],
                            pt2=last_point,
                            color=color,
                            thickness=line_thickness,
                        )
                    except KeyError or IndexError:
                        cv2.circle(
                            img=img,
                            center=int_box[0:2],
                            radius=line_thickness*2,
                            color=color,
                            thickness=-1,
                        )

                if each_subframe == list(result_frame.keys())[0]:
                    # draw class name and index
                    cv2.putText(
                        img,
                        'class:' + str(i_class),
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness
                    )

                    txt_w, txt_h = cv2.getTextSize(
                        str(i_class),
                        fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=text_scale, thickness=text_thickness
                    )

                    cv2.putText(
                        img,
                        'id:' + id_text,
                        (int(x1), int(y1) - 2 * txt_h),
                        cv2.FONT_HERSHEY_PLAIN,
                        text_scale,
                        (0, 255, 255),  # cls_id: yellow
                        thickness=text_thickness
                    )
        result_last_subframe = result_subframe[0]

    return img


def write_results_to_text(
        output_dir: str,
        results_dict: dict,
        data_type: E_text_result_type):
    """
    :param output_dir:
    :param results_dict:
    :param data_type:
    :return:
    """
    try:
        save_format = Dict_text_result_format[data_type]
    except KeyError as e:
        raise e

    file_dir = os.path.join(output_dir, Dict_text_result_name[data_type])

    last_line = None
    with open(file_dir, 'a') as f:
        for frame, result_subframe in results_dict.items():
            if frame > 0:
                for subframe, result_and_fps in result_subframe.items():
                    result_class = result_and_fps[0]
                    fps = result_and_fps[1]
                    for i_class, result_id in result_class.items():
                        for i_id, result in result_id.items():
                            tlwh = result[0]
                            score = result[1]
                            # if data_type == 'kitti':
                            #     i_id -= 1
                            x1, y1, w, h = tlwh
                            # x2, y2 = x1 + w, y1 + h
                            line = save_format.format(
                                frame=frame,
                                subframe=subframe,
                                cls_id=i_class,
                                id=i_id,
                                x1=x1, y1=y1, x2=w, y2=h,
                                score=score,  # detection score
                                fps=fps
                            )
                            if last_line != line:
                                f.write(line)


def write_results_to_video(result_root, frame_dir, video_type, frame_rate):
    output_video_path = os.path.join(result_root, Str_video_result_name)
    a = os.listdir(frame_dir)
    ini_img_dir = os.path.join(frame_dir, os.listdir(frame_dir)[0])
    ini_img = cv2.imread(ini_img_dir)
    res = (ini_img.shape[1], ini_img.shape[0])
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*video_type), frame_rate, res)
    img_list = os.listdir(frame_dir)
    img_list.sort()
    for each_img in img_list:
        if each_img.endswith('.jpg'):
            img_dir = os.path.join(frame_dir, each_img)
            img = cv2.imread(img_dir)
            video.write(img)
    video.release()

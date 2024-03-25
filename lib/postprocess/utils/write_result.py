import os
import cv2
import numpy as np
import typing as typ

Track_Result_Format = typ.Dict[  # track result
                int, typ.Dict[  # class:
                    int, typ.Tuple[  # id:
                        typ.Tuple, float  # tlwh or centerxy, score
                    ]
                ]
            ]

Total_Result_Format = typ.Dict[
    int, typ.Dict[  # frame:
        int, typ.Tuple[  # subframe:
            Track_Result_Format, float  # fps
        ]
    ]
]


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
        result_by_class: Track_Result_Format,
        frame: int,
        fps: float,
):
    """
    :rtype:
    :param image:
    :param result_by_class:
    :param frame:
    :param fps:
    :return:
    """

    img = np.ascontiguousarray(np.copy(image))

    text_scale = max(1.0, image.shape[1] / 1200.)  # 1600.
    text_thickness = 2  # 自定义ID文本线宽
    line_thickness = max(1, int(image.shape[1] / 500.))

    cv2.putText(
        img,
        'frame: %d fps: %.2f' % (frame, fps),
        (0, int(15 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        (0, 0, 255),
        thickness=2
    )

    for i_class, result_by_id in result_by_class.items():
        for i_id, result in result_by_id.items():
            tlwh, score = result

            x1, y1, w, h = tlwh
            int_box = tuple(map(int, (x1, y1, x1 + w, y1 + h)))  # x1, y1, x2, y2
            id_text = '{}'.format(i_id)

            color = get_color(i_class, i_id)

            # draw bbox
            cv2.rectangle(
                img=img,
                pt1=int_box[0:2],  # (x1, y1)
                pt2=int_box[2:4],  # (x2, y2)
                color=color,
                thickness=line_thickness
                          )

            # draw class name and index
            cv2.putText(
                img,
                str(i_class),
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
                id_text,
                (int(x1), int(y1) - txt_h),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 255, 255),  # cls_id: yellow
                thickness=text_thickness
            )

    return img


def write_results_to_text(file_name, results_dict: Total_Result_Format, data_type):
    """
    :param file_name:
    :param results_dict:
    :param data_type:
    :return:
    """
    if data_type == 'mot':
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    elif data_type == 'raw':
        save_format = '{frame},{subframe},{cls_id},{id},{x1},{y1},{w},{h},{score},{fps}\n'
    else:
        raise ValueError(data_type)

    last_line = None
    with open(file_name, 'w') as f:
        for frame, result_subframe in results_dict.items():
            for subframe, result_and_fps in result_subframe.items():
                result_class = result_and_fps[0]
                fps = result_and_fps[1]
                for i_class, result_id in result_class.items():
                    for i_id, result in result_id.items():
                        tlwh = result[0]
                        score = result[1]
                        if data_type == 'kitti':
                            i_id -= 1
                        x1, y1, w, h = tlwh
                        # x2, y2 = x1 + w, y1 + h
                        # line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                        line = save_format.format(
                            frame=frame,
                            subframe=subframe,
                            cls_id=i_class,
                            id=i_id,
                            x1=x1, y1=y1, w=w, h=h,
                            score=score,  # detection score
                            fps=fps
                        )
                        if last_line != line:
                            f.write(line)


def write_results_to_video(result_root, frame_dir, video_type, frame_rate):
    output_video_path = os.path.join(result_root, 'video_result.mp4')
    ini_img = cv2.imread(os.path.join(frame_dir, os.listdir(frame_dir)[0]))
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

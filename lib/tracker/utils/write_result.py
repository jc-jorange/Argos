import os
import cv2


def write_results_to_text(file_name, results_dict, data_type, num_classes=5):
    """
    :param file_name:
    :param results_dict:
    :param data_type:
    :param num_classes:
    :return:
    """
    if data_type == 'mot':
        # save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,{cls_id},1\n'
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score},{cls_id},1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):  # process each object class
            cls_results = results_dict[cls_id]
            for frame_id, tlwhs, track_ids, scores in cls_results:
                if data_type == 'kitti':
                    frame_id -= 1

                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue

                    x1, y1, w, h = tlwh
                    # x2, y2 = x1 + w, y1 + h
                    # line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                    line = save_format.format(frame=frame_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              score=score,  # detection score
                                              cls_id=cls_id)
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

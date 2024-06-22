import os
import glob
import cv2
from collections import defaultdict

from ._masterclass import BasePost
from src.multiprocess_pipeline.workers.postprocess.utils import write_result as wr


class TextResultWriter(BasePost):
    def __init__(
            self,
            save_dir: str,
            result_name: str,
            result: wr.S_default_save,
            dtype: wr.E_text_result_type,
    ):
        super().__init__()
        self.save_dir = os.path.join(save_dir, result_name)
        self.result = result
        self.data_type = dtype

    def process(self):
        wr.write_results_to_text(
            self.save_dir,
            self.result,
            self.data_type
        )


class ImageResultWriter(BasePost):
    def __init__(
            self,
            input_dir: str,
            save_dir: str,
            result_dir: str,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.results_dir = result_dir

        self.results = None

    def process_results(self):
        result_id = defaultdict(tuple)
        result_cls = defaultdict(dict)
        result_subframe = defaultdict(tuple)
        result_frame = defaultdict(dict)

        last_clas = -1
        last_subframe = -1
        last_frame = -1

        with open(self.results_dir, 'r') as f:
            lines = f.readlines()
            for each_line in lines:
                results_raw = each_line.split(',')
                frame = int(results_raw[0])
                subframe = int(results_raw[1])
                cls = int(results_raw[2])
                track_id = int(results_raw[3])
                x1 = float(results_raw[4])
                y1 = float(results_raw[5])
                x2 = float(results_raw[6])
                y2 = float(results_raw[7])
                score = float(results_raw[8])
                fps = float(results_raw[9])

                if frame != last_frame:
                    result_subframe = defaultdict(tuple)
                    last_frame = frame
                    last_subframe = -1
                    last_clas = -1
                if subframe != last_subframe:
                    result_cls = defaultdict(dict)
                    last_subframe = subframe
                    last_clas = -1
                if cls != last_clas:
                    result_id = defaultdict(tuple)
                    last_clas = cls

                result_id[track_id] = ((x1, y1, x2, y2), score)
                result_cls[cls] = result_id
                result_subframe[subframe] = (result_cls, fps)
                result_frame[frame] = result_subframe

        self.results: {} = result_frame

    def process(self):
        self.process_results()

        image_format = ['.jpg', '.jpeg', '.png', '.tif', '.exr']
        image_files = sorted(glob.glob('%s/*.*' % self.input_dir))
        image_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, image_files))

        for each_file in image_files:
            img_0 = cv2.imread(each_file)
            file_name = os.path.splitext(os.path.split(each_file)[1])[0]
            frame = int(file_name)
            if frame in self.results.keys():
                result_frame = self.results[frame]

                img_0 = wr.plot_tracks(img_0, result_frame, frame)

            cv2.imwrite(
                os.path.join(self.save_dir, '{:05d}.jpg'.format(frame)), img_0
            )


class VideoResultWriter(BasePost):
    def __init__(
            self,
            frame_dir: str,
            save_dir: str,
            frame_rate=24,
    ):
        super().__init__()
        self.frame_dir = frame_dir
        self.save_dir = save_dir
        self.frame_rate = frame_rate

    def process(self):
        # write results to video
        wr.write_results_to_video(
            self.save_dir,
            self.frame_dir,
            ('m', 'p', '4', 'v'),
            self.frame_rate
        )

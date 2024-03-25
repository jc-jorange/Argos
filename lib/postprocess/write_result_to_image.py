import os
import glob
import cv2

from lib.postprocess import BasePost
from lib.postprocess.utils import write_result as wr


class TextResultWriter(BasePost):
    def __init__(
            self,
            save_dir: str,
            result_name: str,
            result: wr.Total_Result_Format,
            dtype='mot',
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
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
            result: wr.Total_Result_Format,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.input_dir = input_dir
        self.save_dir = save_dir
        self.result = result

    def process(self):
        image_format = ['.jpg', '.jpeg', '.png', '.tif', '.exr']
        image_files = sorted(glob.glob('%s/*.*' % self.input_dir))
        image_files = list(filter(lambda x: os.path.splitext(x)[1].lower() in image_format, image_files))

        for each_file in image_files:
            img_0 = cv2.imread(each_file)
            file_name = os.path.splitext(os.path.split(each_file)[1])[0]
            frame = int(file_name)
            for each_subframe, result_and_fps in self.result[frame].items():
                result = result_and_fps[0]
                fps = result_and_fps[1]

                result_image = wr.plot_tracks(img_0, result, frame, fps)

                cv2.imwrite(
                    os.path.join(self.save_dir, '{:05d}_{:05d}.jpg'.format(frame, each_subframe)),
                    result_image
                )


class VideoResultWriter(BasePost):
    def __init__(
            self,
            frame_dir: str,
            save_dir: str,
            frame_rate=24,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
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

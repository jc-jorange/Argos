import time
from typing import Type

from ..multiprocess import BaseProcess, ESharedDictType, EMultiprocess
from lib.multiprocess.MP_Tracker import ETrackInfo
from lib.multiprocess.MP_ImageLoader import EImageInfo
from lib.predictor import BasePredictor
from lib.predictor.spline.hermite_spline import HermiteSpline
from lib.tracker.utils.utils import *


class PathPredictProcess(BaseProcess):
    prefix = 'Argus-SubProcess-PathPredictProcess_'

    def __init__(self,
                 predictor_class: Type[BasePredictor],
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.making_process_main_save_dir('camera_predict_')

        self.track_result = None
        self.predict_result = None

        self.predictor = predictor_class()

    def run_begin(self) -> None:
        super().run_begin()
        self.set_logger_file_handler(self.name + '_PathPredict_Log', self.main_output_dir)
        self.logger.info("This is the Path Predictor Process No.{:d}".format(self.idx))

    def run_action(self) -> None:
        super(PathPredictProcess, self).run_action()
        t1 = time.perf_counter()
        i = 0
        self.predictor.time_0 = t1
        while True:
            t2 = time.perf_counter()
            self.track_result = self.container_shared_dict[ESharedDictType.Track][self.idx].read(ETrackInfo.Result)
            if isinstance(self.track_result, np.ndarray):
                self.predict_result = self.predictor.set_new_base(self.track_result)
            else:
                self.predict_result = self.predictor.get_predicted_position(t2)

            image = self.container_shared_dict[ESharedDictType.Image_Current][self.idx].read(EImageInfo.Data)

            if isinstance(self.predict_result, np.ndarray) and isinstance(image, np.ndarray):
                for i_c, e_c in enumerate(self.predict_result):
                    for i_id, e_id in enumerate(e_c):
                        if e_id[0] > 0 and e_id[2] > 0:
                            e_id = e_id.tolist()
                            cv2.circle(image, (int(e_id[0]), int(e_id[1])), 5, (0, 0, 255), -1)
                            cv2.putText(image, 'class:{}, id:{}'.format(i_c, i_id),
                                        (int(e_id[0]) + 10, int(e_id[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (0, 0, 255), 2)

                i += 1
                cv2.imwrite(os.path.join(self.main_output_dir, '{:05d}.jpg'.format(i)), image)

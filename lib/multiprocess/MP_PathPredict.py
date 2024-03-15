import time

from ..multiprocess import BaseProcess, E_SharedDictType, E_Multiprocess
from lib.multiprocess.MP_Tracker import E_TrackInfo
from lib.multiprocess.MP_ImageReceiver import E_ImageInfo
from lib.predictor.spline.hermite_spline import HermiteSpline
import lib.multiprocess.Shared as Sh
from lib.tracker.utils.utils import *
from lib.tracker.utils import write_result as wr, visualization as vis

class PathPredictProcess(BaseProcess):
    prefix = 'Argus-SubProcess-PathPredictProcess_'
    def __init__(self,
                 *args
                 ):
        super().__init__(*args)
        self.making_process_main_save_dir('camera_predict_')

        self.track_result = None
        self.predict_result = None

        self.predictor = HermiteSpline()

    def run(self):
        super(PathPredictProcess, self).run()
        t1 = time.perf_counter()
        i = 0
        self.predictor.time_0 = t1
        while True:
            t2 = time.perf_counter()
            self.track_result = self.container_shared_dict[E_SharedDictType.Track][self.idx].read_data(E_TrackInfo.Result)
            if isinstance(self.track_result, np.ndarray):
                self.predict_result = self.predictor.set_new_base(self.track_result)
            else:
                self.predict_result = self.predictor.get_predicted_position(t2)

            image = self.container_shared_dict[E_SharedDictType.Image][self.idx].read_data(E_ImageInfo.Data)

            if isinstance(self.predict_result, np.ndarray) and isinstance(image, np.ndarray):
                for i_c, e_c in enumerate(self.predict_result):
                    for i_id, e_id in enumerate(e_c):
                        if e_id[0]>0 and e_id[2]>0:
                            e_id = e_id.tolist()
                            cv2.circle(image, (int(e_id[0]), int(e_id[1])), 5, (0, 0, 255), -1)
                            cv2.putText(image, 'class:{}, id:{}'.format(i_c, i_id),
                                        (int(e_id[0])+10, int(e_id[1])-10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.65, (0,0,255),2)

                i += 1
                cv2.imwrite(os.path.join(self.main_output_dir, '{:05d}.jpg'.format(i)), image)

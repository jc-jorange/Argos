from ..multiprocess import BaseProcess, ESharedDictType, EMultiprocess
from lib.matchor.MultiCameraMatch import CenterRayIntersect


class PostIdMatchProcess(BaseProcess):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        ...

    def get_base_ids(self):
        ...

    def match_other_predict(self):
        ...

    def run(self):
        super().run()
        for i_camera, each_camera_predict in enumerate(self.container_shared_dict[ESharedDictType.Predict]):
            if i_camera == 0:
                self.get_base_ids()
            else:
                self.match_other_predict()
        ...

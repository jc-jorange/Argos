from ..multiprocess import BaseProcess, ESharedDictType, EMultiprocess
from lib.matchor.MultiCameraMatch import CenterRayIntersect


class GlobalIdMatchProcess(BaseProcess):
    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        ...

    def get_base_ids(self):
        ...

    def match_other_predict(self):
        ...

    def run_action(self) -> None:
        super().run_action()
        for i_camera, each_camera_predict in enumerate(self.container_shared_dict[ESharedDictType.Predict].item()):
            if i_camera == 0:
                self.get_base_ids()
            else:
                self.match_other_predict()

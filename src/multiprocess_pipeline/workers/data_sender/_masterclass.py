import numpy as np
import time


class BaseDataSender:
    def __init__(self,
                 target,
                 *args,
                 **kwargs):
        self.target = target

        self.target_object = None
        self.count = 0
        self.timestamp = 0

    def get_current_timestamp(self) -> int:
        self.timestamp = int(round(time.time() * 1000))
        return self.timestamp

    def send_data(self, data: np.ndarray) -> bool:
        return False

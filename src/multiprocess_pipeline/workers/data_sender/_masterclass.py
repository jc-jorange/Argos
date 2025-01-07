import numpy as np
import time


class BaseDataSender:
    def __init__(self,
                 target,
                 with_flag=True,
                 ):
        self.target = target
        self.bWith_Flag = with_flag

        self.target_object = None
        self.count = 0

    def send_action(self, timestamp: int, data: np.ndarray) -> bool:
        return False

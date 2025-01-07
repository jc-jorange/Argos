import numpy as np


class BaseCameraTransLoader:
    def __init__(self,
                 source,
                 with_flag=True,
                 ):
        self.source = source
        self.bWith_Flag = with_flag

        self.len = 0
        self.count = 0

    def read_trans(self, idx) -> (int, str, np.ndarray):
        timestamp, trans_path, trans = self.read_action(idx)
        if isinstance(trans, np.ndarray):
            return timestamp, trans_path, trans
        elif isinstance(trans, dict) and trans:
            return timestamp, trans_path, trans
        else:
            if self.count >= len(self):
                raise StopIteration
            else:
                self.__next__()

    def read_action(self, idx) -> (int, str, np.ndarray):
        return 0, '', np.ndarray

    def pre_process(self) -> bool:
        return True

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        timestamp, path, trans = self.read_trans(self.count)
        if trans:
            return timestamp, path, trans
        else:
            raise StopIteration

    def __getitem__(self, idx):
        idx = idx % len(self)
        return self.read_trans(idx)

    def __len__(self):
        return self.len


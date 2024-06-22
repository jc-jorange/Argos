import numpy as np


class BaseCameraTransLoader:
    def __init__(self,
                 source,
                 *args,
                 **kwargs
                 ):
        self.source = source

        self.len = 0
        self.count = 0

    def read_trans(self, idx) -> (int, str, np.ndarray):
        return 0, '', np.ndarray

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


import os.path
import numpy as np

from ._masterclass import BaseCameraTransLoader


class CamTransFileLoader(BaseCameraTransLoader):
    def __init__(self,
                 *args,
                 **kwargs):
        super(CamTransFileLoader, self).__init__(*args, **kwargs)

        self.timestamp = []
        self.trans = []

        if os.path.isfile(self.source):
            timestamp_format = ['.csv', '.txt']
            if os.path.splitext(self.source)[-1] in timestamp_format:
                with open(self.source) as f:
                    lines = f.readlines()
                    for each_result in lines:
                        each_result = each_result.strip()
                        each_result = each_result.split(',')
                        timestamp = each_result[0]
                        trans = each_result[1]

                        self.timestamp.append(timestamp)
                        self.trans.append(trans)

        self.len = len(self.trans)
        assert self.len > 0, f'No Camera Transform found in {self.source}'

    def read_action(self, idx) -> (int, str, np.ndarray):
        super(CamTransFileLoader, self).read_action(idx)

        trans = self.trans[idx]
        trans = np.asarray(trans)
        timestamp = self.timestamp[idx]

        return timestamp, self.source, trans

    def __next__(self):
        super(CamTransFileLoader, self).__next__()
        if self.count == len(self):
            raise StopIteration
        return self.read_action(self.count - 1)

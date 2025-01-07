import numpy as np
from typing import Tuple

from ._masterclass import BaseDataFilter


class SpecialIDDataFilter(BaseDataFilter):
    def __init__(self,
                 target: Tuple[int, int],
                 *args,
                 **kwargs):
        super(SpecialIDDataFilter, self).__init__(*args, **kwargs)
        self.target = target

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        none_zero = np.nonzero(data)
        result = None
        if self.target[0] in none_zero[0] and self.target[1] in none_zero[1]:
            result = data[self.target]
        return result

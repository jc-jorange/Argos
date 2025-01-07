import numpy as np

from ._masterclass import BaseDataFilter


class FirstValidDataFilter(BaseDataFilter):
    def __init__(self,
                 *args,
                 **kwargs):
        super(FirstValidDataFilter, self).__init__(*args, **kwargs)

    def filter_data(self, data: np.ndarray) -> np.ndarray:
        none_zero = np.nonzero(data)
        result = None
        if len(none_zero[0]):
            f_class = none_zero[0][0]
            f_id = none_zero[1][0]
            none_zero_id = (none_zero[0][0], none_zero[1][0])
            result = np.copy(data)
            result.fill(0)
            result[f_class][f_id] = data[f_class][f_id]
        return result

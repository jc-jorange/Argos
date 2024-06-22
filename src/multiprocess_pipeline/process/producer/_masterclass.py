import ctypes
from multiprocessing import Value

from .._masterclass import BaseProcess


class ProducerProcess(BaseProcess):
    def __init__(self,
                 *args, **kwargs) -> None:
        super(ProducerProcess, self).__init__(*args, **kwargs,)
        self._b_receive = Value(ctypes.c_bool, True)

    def recv_alive(self) -> bool:
        return self._b_receive.value

    def run_end(self) -> None:
        super(ProducerProcess, self).run_end()
        self._b_receive.value = False

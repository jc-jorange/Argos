from .._masterclass import BaseProcess


class LogProcess(BaseProcess):
    def __init__(self,
                 *args, **kwargs) -> None:
        super(LogProcess, self).__init__(*args, **kwargs,)
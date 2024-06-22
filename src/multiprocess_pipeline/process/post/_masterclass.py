from .._masterclass import BaseProcess


class PostProcess(BaseProcess):
    def __init__(self,
                 *args, **kwargs) -> None:
        super(PostProcess, self).__init__(*args, **kwargs,)

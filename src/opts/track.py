import argparse

from ._base_opt import opts

from src.multiprocess_pipeline.utils import check as pipe_check


class opts_track(opts):
    def __init__(self):
        super(opts_track, self).__init__()
        self.parser: argparse.ArgumentParser

        # basic experiment setting
        self.parser.add_argument('--pipeline_cfg',
                                 type=str,
                                 default='./src/multiprocess_pipeline/cfg/Predict.yml',
                                 help='load multi-process pipeline structure from cfg')
        # visualization
        self.parser.add_argument('--allow_show_image',
                                 action='store_true',
                                 help='whether allow to image data in realtime')

        # track
        self.parser.add_argument('--realtime',
                                 action='store_true',
                                 help='whether read image data in realtime')

        self.parser.add_argument('--half_precision',
                                 action='store_true',
                                 help='make model into half precision mode(f16)')

        self.parser.add_argument('--cuda_stream',
                                 action='store_true',
                                 help='enable pytorch cuda stream for parallelling')

        self.parser.add_argument('--quantization',
                                 action='store_true',
                                 help='enable pytorch quantization for torch.nn.Linear')

        self.parser.add_argument('--GPU_load_sequence',
                                 action='store_true',
                                 help='force all GPU heavy load execute sequentially')

    def parse(self, args=''):
        super(opts_track, self).parse(args=args)

        self._mk_save_dir('run_result')

        return self._opt

    def init(self, args=''):
        super(opts_track, self).init(args=args)

        pipe_check.check_pipeline_cfg(self._opt.pipeline_cfg)

        return self._opt

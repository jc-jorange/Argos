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

        # track
        self.parser.add_argument('--realtime',
                                 type=bool,
                                 default=False,
                                 help='whether read image data in realtime')

    def parse(self, args=''):
        super(opts_track, self).parse(args=args)

        self._mk_save_dir('run_result')

        return self._opt

    def init(self, args=''):
        super(opts_track, self).init(args=args)

        pipe_check.check_pipeline_cfg(self._opt.pipeline_cfg)

        return self._opt

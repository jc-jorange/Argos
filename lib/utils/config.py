from yacs import CfgNode

cfg = CfgNode(new_allowed=True)

cfg.save_dir = "./"

# NETWORK params
cfg.model = CfgNode(new_allowed=True)
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET params
cfg.dataset = CfgNode(new_allowed=True)
cfg.dataset.train = CfgNode(new_allowed=True)
cfg.dataset.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)

cfg.log = CfgNode(new_allowed=True)


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    import sys

    with open(sys.argv[1], "w") as f:
        print(cfg, file=f)

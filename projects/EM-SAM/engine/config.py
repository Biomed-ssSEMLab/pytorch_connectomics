from yacs.config import CfgNode as CN

def add_P95_config(cfg):
    cfg.P95 = CN()
    cfg.p95.model = "mae"
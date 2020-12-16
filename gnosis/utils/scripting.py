import random
from upcycle.random.seed import set_all_seeds
import hydra
from omegaconf import OmegaConf, DictConfig
from tensorboardX import SummaryWriter
import torch


def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)
    tb_logger = SummaryWriter(log_dir=".")

    print(hydra_cfg.pretty())
    with open('hydra_config.txt', 'w') as f:
        f.write(hydra_cfg.pretty())
    tb_logger.add_text("hypers/hydra_cfg", hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")
    return hydra_cfg, logger, tb_logger
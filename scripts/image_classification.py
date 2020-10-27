import hydra
import pandas as pd
import random
import torch

from omegaconf import OmegaConf, DictConfig
from gnosis.boilerplate import train_loop
from gnosis.utils.data import get_loaders

from upcycle.random.seed import set_all_seeds
from upcycle.cuda import try_cuda


def startup(hydra_cfg):
    if hydra_cfg.seed is None:
        seed = random.randint(0, 100000)
        hydra_cfg['seed'] = seed
        set_all_seeds(seed)

    logger = hydra.utils.instantiate(hydra_cfg.logger)
    hydra_cfg = OmegaConf.to_container(hydra_cfg, resolve=True)  # Resolve config interpolations
    hydra_cfg = DictConfig(hydra_cfg)
    logger.write_hydra_yaml(hydra_cfg)

    print(hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")

    return hydra_cfg, logger


@hydra.main(config_path='../hydra', config_name='image_classification')
def main(config):
    # construct logger, model, dataloaders
    config, s3_logger = startup(config)
    model = hydra.utils.instantiate(config.model)
    model = try_cuda(model)
    trainloader, testloader = get_loaders(config)

    print("==== training the network ====")
    train_loop(
        config,
        model,
        trainloader=trainloader,
        testloader=testloader,
        s3_logger=s3_logger
    )


if __name__ == '__main__':
    main()

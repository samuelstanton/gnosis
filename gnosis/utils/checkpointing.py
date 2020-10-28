import pickle as pkl
from pathlib import Path
import yaml
import os
import hydra

from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_models(ckpt_dir):
    ckpt_path = os.path.join(hydra.utils.get_original_cwd(), ckpt_dir)
    ckpt_path = os.path.normpath(ckpt_path)
    ckpt_path = Path(ckpt_path)
    config_files = list(ckpt_path.rglob('*/.hydra/config.yaml'))
    ckpt_files = list(ckpt_path.rglob('*/teacher_*.ckpt'))

    if len(config_files) == 0:
        print(f"no model config files found in {ckpt_dir}")
        return []
    if len(ckpt_files) == 0:
        print(f"no checkpoints found in {ckpt_dir}")
        return []

    saved_models = []
    for cfg_file, ckpt_file in zip(config_files, ckpt_files):
        config = OmegaConf.load(cfg_file)
        model = instantiate(config.model)

        with open(ckpt_file, 'rb') as f:
            state_dict = pkl.load(f)
        model.load_state_dict(state_dict)
        saved_models.append(model)

    return saved_models
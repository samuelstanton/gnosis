import pickle as pkl
from pathlib import Path
import os
import hydra

from hydra.utils import instantiate
from omegaconf import OmegaConf
from upcycle.cuda import try_cuda


def load_teachers(ckpt_dir):
    ckpt_path = os.path.join(hydra.utils.get_original_cwd(), ckpt_dir)
    ckpt_path = os.path.normpath(ckpt_path)
    ckpt_path = Path(ckpt_path)
    config_files = list(ckpt_path.rglob('config.yaml'))
    ckpt_files = list(ckpt_path.rglob('teacher_*.ckpt'))

    if len(config_files) == 0:
        print(f"no model config files found in {ckpt_dir}")
        return []
    if len(ckpt_files) == 0:
        print(f"no checkpoints found in {ckpt_dir}")
        return []

    cfg_file = config_files[0]
    saved_models = []
    for ckpt_file in ckpt_files:
        config = OmegaConf.load(cfg_file)
        model = instantiate(config.model)

        with open(ckpt_file, 'rb') as f:
            state_dict = pkl.load(f)
        model.load_state_dict(state_dict)
        saved_models.append(model)
    print('==== teacher checkpoints loaded successfully ====')

    return saved_models


def load_generator(config):
    print('==== loading generator checkpoint ====')
    generator = hydra.utils.instantiate(config.density_model.gen_cfg)
    weight_dir = os.path.join(hydra.utils.get_original_cwd(), config.density_model.gen_weight_dir)
    search_pattern = '*generator_500.ckpt'
    print(f'searching in {weight_dir}')
    ckpt_files = Path(weight_dir).rglob(search_pattern)
    ckpt_files = [f.as_posix() for f in ckpt_files]
    if len(ckpt_files) < 1:
        raise RuntimeError(f'no checkpoints matching {search_pattern} found.')
    with open(ckpt_files[0], 'rb') as f:
        state_dict = pkl.load(f)
    generator.load_state_dict(state_dict)
    print('==== generator checkpoint loaded successfully ====')
    return try_cuda(generator)

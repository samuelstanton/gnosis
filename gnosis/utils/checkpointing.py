import pickle as pkl
from pathlib import Path
import os
import hydra
import yaml

from hydra.utils import instantiate
from omegaconf import OmegaConf
from upcycle.cuda import try_cuda
from upcycle.checkpointing import s3_load_yaml, s3_load_obj


def load_teachers(config, ckpt_pattern='*teacher_?.ckpt', **init_kwargs):
    ckpt_path = os.path.join(config.data_dir, config.teacher.ckpt_dir)
    if config.ckpt_store == 'local':
        proj_dir = get_proj_dir(config)
        ckpt_path = os.path.join(proj_dir, ckpt_path)
        ckpt_path = os.path.normpath(ckpt_path)
        config_ckpts, _ = local_load_yaml(ckpt_path, 'config.yaml')
        weight_ckpts, weight_files = local_load_obj(ckpt_path, ckpt_pattern)
    elif config.ckpt_store == 's3':
        config_ckpts, _ = s3_load_yaml(config.s3_bucket, ckpt_path, '*config.yaml')
        weight_ckpts, weight_files = s3_load_obj(config.s3_bucket, ckpt_path, ckpt_pattern)
    else:
        raise RuntimeError('unrecognized checkpoint store')

    teacher_cfg = OmegaConf.create(config_ckpts[0])
    if 'classifier' not in teacher_cfg.keys():
        teacher_cfg.classifier = teacher_cfg.model  # support old checkpoint configs
    assert teacher_cfg.classifier.depth == config.teacher.depth  # confirm checkpoints are correct depth
    teachers = []
    for state_dict in weight_ckpts:
        model = instantiate(teacher_cfg.classifier, **init_kwargs)
        model.load_state_dict(state_dict)
        teachers.append(try_cuda(model))
    print(f'==== {len(teachers)} teacher checkpoint(s) matching {ckpt_pattern} loaded successfully ====')
    return teachers, weight_files


def load_generator(config):
    data_dir = 'data/experiments/image_generation'
    ckpt_path = os.path.join(data_dir, config.density_model.ckpt_dir)
    if config.ckpt_store == 'local':
        proj_dir = get_proj_dir(config)
        ckpt_path = os.path.join(proj_dir, ckpt_path)
        ckpt_path = os.path.normpath(ckpt_path)
        config_ckpts, _ = local_load_yaml(ckpt_path, 'config.yaml')
        weight_ckpts, weight_files = local_load_obj(ckpt_path, config.density_model.ckpt_pattern)
    elif config.ckpt_store == 's3':
        config_ckpts, _ = s3_load_yaml(config.s3_bucket, ckpt_path, '*config.yaml')
        weight_ckpts, weight_files = s3_load_obj(config.s3_bucket, ckpt_path, config.density_model.ckpt_pattern)
    else:
        raise RuntimeError('unrecognized checkpoint store')

    generator_cfg = OmegaConf.create(config_ckpts[0])
    generators = []
    for state_dict in weight_ckpts:
        model = instantiate(generator_cfg.density_model.gen_cfg)
        model.load_state_dict(state_dict)
        generators.append(try_cuda(model))
    print(f'==== {len(generators)} generator checkpoint(s) loaded successfully ====')
    return generators, weight_files


def get_proj_dir(config):
    if config.project_dir is None:
        return hydra.utils.get_original_cwd()
    if config.project_dir.lower() == 'cwd':
        return hydra.utils.get_original_cwd()
    return config.project_dir


def local_load_yaml(root_dir, glob_pattern):
    files = [f.as_posix() for f in Path(root_dir).rglob(glob_pattern)]
    files.sort()
    if len(files) < 1:
        raise FileNotFoundError
    results = []
    for file in files:
        with open(file, 'r') as f:
            results.append(yaml.full_load(f))
    return results, files


def local_load_obj(root_dir, glob_pattern):
    files = [f.as_posix() for f in Path(root_dir).rglob(glob_pattern)]
    files.sort()
    if len(files) < 1:
        raise FileNotFoundError
    results = []
    for file in files:
        with open(file, 'rb') as f:
            results.append(pkl.load(f))
    return results, files


def select_ckpts(ckpts, trial_id, num_elements, ckpt_names=None):
    start_idx = (trial_id * num_elements) % len(ckpts) - len(ckpts)
    stop_idx = start_idx + num_elements
    print(f'using checkpoints {[(len(ckpts) + i) % len(ckpts) for i in range(start_idx, stop_idx)]}')

    if ckpt_names is not None:
        assert len(ckpt_names) == len(ckpts)
        for i in range(start_idx, stop_idx):
            print(f'{(len(ckpts) + i) % len(ckpts)}: {ckpt_names[i]}')

    return [ckpts[i] for i in range(start_idx, stop_idx)]

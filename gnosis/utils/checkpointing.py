import pickle as pkl
from pathlib import Path
import os
import hydra
import yaml

from hydra.utils import instantiate
from omegaconf import OmegaConf
from upcycle.cuda import try_cuda
from upcycle.checkpointing import s3_load_yaml, s3_load_obj


def load_teachers(config):
    if config.ckpt_store == 'local':
        proj_dir = get_proj_dir(config)
        ckpt_path = os.path.join(proj_dir, config.teacher.ckpt_dir)
        ckpt_path = os.path.normpath(ckpt_path)
        config_ckpts = local_load_yaml(ckpt_path, 'config.yaml')
        weight_ckpts = local_load_obj(ckpt_path, 'teacher_*.ckpt')
    elif config.ckpt_store == 's3':
        config_ckpts = s3_load_yaml(config.s3_bucket, config.teacher.ckpt_dir, '*config.yaml')
        weight_ckpts = s3_load_obj(config.s3_bucket, config.teacher.ckpt_dir, '*teacher_*.ckpt')
    else:
        raise RuntimeError('unrecognized checkpoint store')

    teacher_cfg = OmegaConf.create(config_ckpts[0])
    if 'classifier' not in teacher_cfg.keys():
        teacher_cfg.classifier = teacher_cfg.model  # support old checkpoint configs
    assert teacher_cfg.classifier.depth == config.teacher.depth  # confirm checkpoints are correct depth
    teachers = []
    for state_dict in weight_ckpts:
        model = instantiate(teacher_cfg.classifier)
        model.load_state_dict(state_dict)
        teachers.append(try_cuda(model))
    print(f'==== {len(teachers)} teacher checkpoint(s) loaded successfully ====')
    return teachers


def load_generator(config):
    if config.ckpt_store == 'local':
        proj_dir = get_proj_dir(config)
        ckpt_path = os.path.join(proj_dir, config.density_model.weight_dir)
        ckpt_path = os.path.normpath(ckpt_path)
        config_ckpts = local_load_yaml(ckpt_path, 'config.yaml')
        weight_ckpts = local_load_obj(ckpt_path, '*generator_500.ckpt')
    elif config.ckpt_store == 's3':
        config_ckpts = s3_load_yaml(config.s3_bucket, config.density_model.weight_dir, '*config.yaml')
        weight_ckpts = s3_load_obj(config.s3_bucket, config.density_model.weight_dir, '*generator_500.ckpt')
    else:
        raise RuntimeError('unrecognized checkpoint store')

    generator_cfg = OmegaConf.create(config_ckpts[0])
    generators = []
    for state_dict in weight_ckpts:
        model = instantiate(generator_cfg.density_model.gen_cfg)
        model.load_state_dict(state_dict)
        generators.append(try_cuda(model))
    print(f'==== {len(generators)} generator checkpoint(s) loaded successfully ====')
    return generators

    # print('==== loading generator checkpoint ====')
    # generator = hydra.utils.instantiate(config.density_model.gen_cfg)
    # proj_dir = get_proj_dir(config)
    # weight_dir = os.path.join(proj_dir, config.density_model.gen_weight_dir)
    # search_pattern = '*generator_500.ckpt'
    # print(f'searching in {weight_dir}')
    # ckpt_files = Path(weight_dir).rglob(search_pattern)
    # ckpt_files = [f.as_posix() for f in ckpt_files]
    # if len(ckpt_files) < 1:
    #     raise RuntimeError(f'no checkpoints matching {search_pattern} found.')
    # with open(ckpt_files[0], 'rb') as f:
    #     state_dict = pkl.load(f)
    # generator.load_state_dict(state_dict)
    # print('==== generator checkpoint loaded successfully ====')
    # return try_cuda(generator)


def get_proj_dir(config):
    if config.project_dir is None:
        return hydra.utils.get_original_cwd()
    if config.project_dir.lower() == 'cwd':
        return hydra.utils.get_original_cwd()
    return config.project_dir


def local_load_yaml(root_dir, glob_pattern):
    files = [f.as_posix() for f in Path(root_dir).rglob(glob_pattern)]
    if len(files) < 1:
        raise FileNotFoundError
    results = []
    for file in files:
        with open(file, 'r') as f:
            results.append(yaml.full_load(f))
    return results


def local_load_obj(root_dir, glob_pattern):
    files = [f.as_posix() for f in Path(root_dir).rglob(glob_pattern)]
    if len(files) < 1:
        raise FileNotFoundError
    results = []
    for file in files:
        with open(file, 'rb') as f:
            results.append(pkl.load(f))
    return results

import hydra
import pandas as pd
import os
import random
import torch

from gnosis import distillation, models
from omegaconf import OmegaConf, DictConfig
from gnosis.boilerplate import train_loop
from gnosis.utils.data import get_loaders
from gnosis.utils.checkpointing import load_models

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
    config, logger = startup(config)
    trainloader, testloader = get_loaders(config)

    teachers = load_models(config.teacher.ckpt_dir)
    if len(teachers) >= config.teacher.num_components and config.teacher.use_ckpts is True:
        teachers = teachers[:config.teacher.num_components]
        teachers = try_cuda(*teachers)
    else:
        teachers = []
        for i in range(config.teacher.num_components):
            model = hydra.utils.instantiate(config.model)
            model = try_cuda(model)
            teacher_loss = distillation.ClassifierTeacherLoss(model)
            print(f"==== training teacher model {i+1} ====")
            model, records = train_loop(config, model, teacher_loss, trainloader, testloader)
            teachers.append(model)

            logger.add_table(f'teacher_{i}_train_metrics', records)
            logger.write_csv()
            logger.save_obj(model.state_dict(), f'teacher_{i}.ckpt')

    teacher = models.ClassifierEnsemble(*teachers)
    student = hydra.utils.instantiate(config.model)
    student = try_cuda(student)
    student_loss = distillation.ClassifierStudentLoss(teacher, student)
    print(f"==== training the student model ====")
    student, records = train_loop(config, student, student_loss, trainloader, testloader)
    logger.add_table(f'student_train_metrics', records)
    logger.write_csv()
    logger.save_obj(student.state_dict(), f'student.ckpt')


if __name__ == '__main__':
    main()

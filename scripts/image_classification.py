import hydra
import pandas as pd
import os
import random
import torch
from tensorboardX import SummaryWriter

from gnosis import distillation, models
from omegaconf import OmegaConf, DictConfig
from gnosis.boilerplate import train_loop, eval_epoch
from gnosis.utils.data import get_loaders, get_generator
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
    tb_logger = SummaryWriter(log_dir=".")

    print(hydra_cfg.pretty())
    with open('hydra_config.txt', 'w') as f:
        f.write(hydra_cfg.pretty())
    tb_logger.add_text("hypers/hydra_cfg", hydra_cfg.pretty())
    print(f"GPU available: {torch.cuda.is_available()}")
    return hydra_cfg, logger, tb_logger


@hydra.main(config_path='../hydra', config_name='image_classification')
def main(config):
    # construct logger, model, dataloaders
    config, logger, tb_logger = startup(config)
    trainloader, testloader = get_loaders(config)
    tb_logger.add_text("hypers/transforms", config.augmentation.transforms_list, 0)

    teachers = load_models(config.teacher.ckpt_dir)
    if len(teachers) >= config.teacher.num_components and config.teacher.use_ckpts is True:
        teachers = [try_cuda(teachers[i]) for i in range(config.teacher.num_components)]
    else:
        teachers = []
        for i in range(config.teacher.num_components):
            model = hydra.utils.instantiate(config.model)
            model = try_cuda(model)
            teacher_loss = distillation.ClassifierTeacherLoss(model)
            print(f"==== training teacher model {i+1} ====")

            tb_prefix = "teachers/teacher_{}/".format(i)
            model, records = train_loop(config, model, teacher_loss, trainloader, testloader, tb_logger, tb_prefix)
            teachers.append(model)

            logger.add_table(f'teacher_{i}_train_metrics', records)
            logger.write_csv()
            logger.save_obj(model.state_dict(), f'teacher_{i}.ckpt')

    print('==== ensembling teacher models ====')
    teacher = models.ClassifierEnsemble(*teachers)
    _, teacher_train_acc = eval_epoch(teacher, trainloader, models.ensemble.ClassifierEnsembleLoss(teacher))
    _, teacher_test_acc = eval_epoch(teacher, testloader, models.ensemble.ClassifierEnsembleLoss(teacher))

    student = hydra.utils.instantiate(config.model)
    student = try_cuda(student)

    generator = get_generator(config) if config.trainer.generator.enabled else None
    student_base_loss = ydra.utils.instantiate(config.loss.init)
    student_loss = distillation.ClassifierStudentLoss(
        teacher, student, student_base_loss, generator,
        gen_ratio=config.trainer.generator.gen_ratio)
    print(f"==== training the student model ====")
    student, records = train_loop(config, student, student_loss, trainloader, testloader, tb_logger)
    for r in records:
        r.update(dict(teacher_test_acc=teacher_test_acc, teacher_train_acc=teacher_train_acc))
    logger.add_table(f'student_train_metrics', records)
    logger.write_csv()
    logger.save_obj(student.state_dict(), f'student.ckpt')


if __name__ == '__main__':
    main()

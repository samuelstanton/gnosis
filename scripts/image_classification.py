import hydra
from upcycle.cuda import try_cuda
import random

from gnosis import distillation, models
from gnosis.boilerplate import train_loop, eval_epoch
from gnosis.utils.data import get_loaders
from gnosis.utils.checkpointing import load_teachers, load_generator
from gnosis.utils.scripting import startup


@hydra.main(config_path='../config', config_name='image_classification')
def main(config):
    # construct logger, model, dataloaders
    config, logger, tb_logger = startup(config)
    trainloader, testloader = get_loaders(config)
    tb_logger.add_text("hypers/transforms", config.augmentation.transforms_list, 0)

    try:
        teachers = load_teachers(config)
        random.shuffle(teachers)
    except FileNotFoundError:
        teachers = []
    if len(teachers) >= config.teacher.num_components and config.teacher.use_ckpts is True:
        teachers = [try_cuda(teachers[i]) for i in range(config.teacher.num_components)]
    else:
        for i in range(config.teacher.num_components):
            model = hydra.utils.instantiate(config.classifier)
            model = try_cuda(model)
            teacher_loss = distillation.ClassifierTeacherLoss(model)
            print(f"==== training teacher model {i+1} ====")

            tb_prefix = "teachers/teacher_{}/".format(i)
            model, records = train_loop(config, None, model, teacher_loss, trainloader,
                                        testloader, tb_logger, tb_prefix)
            teachers.append(model)

            logger.add_table(f'teacher_{i}_train_metrics', records)
            logger.write_csv()
            logger.save_obj(model.state_dict(), f'teacher_{i}.ckpt')

    if config.trainer.distill_teacher is False:
        return float('NaN')

    print('==== ensembling teacher classifiers ====')
    teacher = models.ClassifierEnsemble(*teachers)
    _, teacher_train_acc = eval_epoch(teacher, trainloader, models.ensemble.ClassifierEnsembleLoss(teacher))
    _, teacher_test_acc = eval_epoch(teacher, testloader, models.ensemble.ClassifierEnsembleLoss(teacher))

    student = hydra.utils.instantiate(config.classifier)
    student = try_cuda(student)

    generator = None
    if config.trainer.synth_aug.enabled and config.trainer.synth_aug.ratio > 0:
        assert config.augmentation.normalization == 'unitcube'  # GANs use Tanh activations when sampling
        generator = load_generator(config)[0]
    student_base_loss = hydra.utils.instantiate(config.loss.init)
    student_loss = distillation.ClassifierStudentLoss(
        teacher, student, student_base_loss, generator,
        gen_ratio=config.trainer.synth_aug.ratio)

    print(f"==== distilling student classifier ====")
    student, records = train_loop(config, teacher, student, student_loss, trainloader, testloader, tb_logger)
    for r in records:
        r.update(dict(teacher_test_acc=teacher_test_acc, teacher_train_acc=teacher_train_acc))
    logger.add_table(f'student_train_metrics', records)
    logger.write_csv()
    logger.save_obj(student.state_dict(), f'student.ckpt')
    return 1 - records[-1]['test_acc'] / 100.


if __name__ == '__main__':
    main()

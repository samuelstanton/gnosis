import hydra
from upcycle.cuda import try_cuda
import random
import math

from gnosis import distillation, models
from gnosis.boilerplate import train_loop, eval_epoch, supervised_epoch, distillation_epoch
from gnosis.utils.data import get_loaders, make_synth_teacher_data, save_logits, get_distill_loaders
from gnosis.utils.checkpointing import load_teachers, load_generator
from gnosis.utils.scripting import startup


@hydra.main(config_path='../config', config_name='image_classification')
def main(config):
    # construct logger, model, dataloaders
    config, logger, tb_logger = startup(config)
    train_loader, test_loader = get_loaders(config)
    tb_logger.add_text("hypers/transforms", ''.join(config.augmentation.transforms_list), 0)

    if config.teacher.use_ckpts:
        try:
            teachers = load_teachers(config)
            if config.teacher.shuffle_ckpts:
                print('shuffling checkpoints')
                random.shuffle(teachers)
        except FileNotFoundError:
            teachers = []
        if len(teachers) >= config.teacher.num_components:
            # use trial_id to determine which checkpoints are used
            start_idx = (config.trial_id * config.teacher.num_components) % len(teachers) - len(teachers)
            stop_idx = start_idx + config.teacher.num_components
            print(f'using checkpoints {[(len(teachers) + i) % len(teachers) for i in range(start_idx, stop_idx)]}')
            teachers = [try_cuda(teachers[i]) for i in range(start_idx, stop_idx)]
    else:
        teachers = []
    num_ckpts = len(teachers)
    # if there weren't enough checkpoints, train the remaining components
    for i in range(num_ckpts, config.teacher.num_components):
        model = hydra.utils.instantiate(config.classifier)
        model = try_cuda(model)
        teacher_loss = distillation.ClassifierTeacherLoss(model)
        print(f"==== training teacher model {i+1} ====")
        model, records = train_loop(
            config,
            model,
            train_closure=supervised_epoch,
            train_loader=train_loader,
            train_kwargs=dict(loss_fn=teacher_loss),
            eval_closure=eval_epoch,
            eval_loader=test_loader,
            eval_kwargs=dict(loss_fn=teacher_loss),
            tb_logger=tb_logger,
            tb_prefix="teacher_{}_".format(i)
        )
        teachers.append(model)
        logger.add_table(f'teacher_{i}_train_metrics', records)
        logger.write_csv()
    for i, model in enumerate(teachers):
        logger.save_obj(model.state_dict(), f'teacher_{i}.ckpt')

    if config.trainer.distill_teacher is False:
        return float('NaN')

    print('==== ensembling teacher classifiers ====')
    teacher = models.ClassifierEnsemble(*teachers)
    teacher_train_metrics = eval_epoch(teacher, train_loader, models.ensemble.ClassifierEnsembleLoss(teacher))
    teacher_test_metrics = eval_epoch(teacher, test_loader, models.ensemble.ClassifierEnsembleLoss(teacher))

    student = hydra.utils.instantiate(config.classifier)
    student = try_cuda(student)

    synth_data = None
    if config.trainer.synth_aug.ratio > 0:
        assert config.augmentation.normalization == 'unitcube'  # GANs use Tanh activations when sampling
        generator = load_generator(config)[0]
        num_synth = math.ceil(len(train_loader.dataset) * config.trainer.synth_aug.ratio)
        print(f'==== generating {num_synth} synthetic examples ====')
        synth_data = make_synth_teacher_data(generator, teacher, num_synth,
                                             batch_size=config.dataloader.batch_size)
        del generator  # free up memory
    train_loader, synth_loader = get_distill_loaders(config, train_loader, synth_data)
    student_base_loss = hydra.utils.instantiate(config.loss.init)
    student_loss = distillation.ClassifierStudentLoss(student, student_base_loss, config.loss.alpha)

    print(f"==== distilling student classifier ====")
    student, records = train_loop(
        config,
        student,
        train_closure=distillation_epoch,
        train_loader=train_loader,
        train_kwargs=dict(loss_fn=student_loss, teacher=teacher, synth_loader=synth_loader),
        eval_closure=eval_epoch,
        eval_loader=test_loader,
        eval_kwargs=dict(loss_fn=student_loss, teacher=teacher),
        tb_logger=tb_logger,
        tb_prefix="student_"
    )

    logger.save_obj(student.state_dict(), f'student.ckpt')
    # for r in records:
    #     r.update({"teacher_train/acc": teacher_train_metrics['test_acc'],
    #               "teacher_test/acc": teacher_test_metrics['test_acc'],
    #               "teacher_test/ece": teacher_test_metrics['test_ece']})
    logger.add_table(f'student_train_metrics', records)
    logger.write_csv()

    del train_loader, test_loader  # these will be regenerated w/o augmentation
    save_logits(config, student, teacher, synth_data, logger)
    res = 1 - records[-1]['test_acc'] / 100. if len(records) > 0 else float('NaN')
    return res


if __name__ == '__main__':
    main()

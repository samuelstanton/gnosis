import hydra
from upcycle.cuda import try_cuda
from functools import partial
import random
import logging
import traceback
import torch

torch.backends.cudnn.enabled = False

from gnosis import distillation, models
from gnosis.boilerplate import train_loop, eval_epoch, supervised_epoch, distillation_epoch
from gnosis.utils.data import get_text_loaders, save_logits, get_distill_loaders
from gnosis.utils.checkpointing import load_teachers, select_ckpts

from upcycle.scripting import startup
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf

from gnosis.utils.initialization import interpolate_net


@hydra.main(config_path='../config', config_name='text_classification')
def main(config):
    try:
        # construct logger, model, dataloaders
        config, logger = startup(config)
        train_loader, test_loader, train_splits, vocab_size = get_text_loaders(config)
        tb_logger = SummaryWriter(log_dir=".")

        if config.teacher.use_ckpts:
            try:
                teachers, ckpt_files = load_teachers(config, num_words=vocab_size)
                if config.teacher.shuffle_ckpts:
                    print('shuffling checkpoints')
                    random.shuffle(teachers)
            except FileNotFoundError:
                teachers = []
            if len(teachers) >= config.teacher.num_components:
                # use trial_id to determine which checkpoints are used
                teachers = select_ckpts(teachers, config.trial_id, config.teacher.num_components, ckpt_names=ckpt_files)
                teachers = [try_cuda(m) for m in teachers]
                # teachers = [try_cuda(teachers[i]) for i in range(start_idx, stop_idx)]
        else:
            teachers = []
        num_ckpts = len(teachers)
        # if there weren't enough checkpoints, train the remaining components
        for i in range(num_ckpts, config.teacher.num_components):
            model = hydra.utils.instantiate(config.classifier, num_words=vocab_size)
            model = try_cuda(model)

            logger.save_obj(model.state_dict(), f'teacher_init_{i}.ckpt')
            print(f"==== training teacher model {i + 1} ====")
            teacher_loss = distillation.ClassifierTeacherLoss(model)
            model, records = train_loop(
                config,
                model,
                train_closure=supervised_epoch,
                train_loader=train_loader,
                train_kwargs=dict(loss_fn=teacher_loss),
                eval_closure=partial(eval_epoch, drop_synthetic_inputs=False),
                eval_loader=test_loader,
                eval_kwargs=dict(loss_fn=teacher_loss),
                tb_logger=tb_logger,
                tb_prefix="teachers/teacher_{}/".format(i)
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
        # distill_splits = [train_splits[i] for i in config.distill_loader.splits]
        distill_loader = hydra.utils.instantiate(config.distill_loader, loader=train_loader, teacher=teacher)
        teacher_train_metrics = eval_epoch(teacher, distill_loader, epoch=0,
                                           loss_fn=models.ensemble.ClassifierEnsembleLoss(teacher),
                                           drop_synthetic_inputs=False)
        teacher_test_metrics = eval_epoch(teacher, test_loader, epoch=0,
                                          loss_fn=models.ensemble.ClassifierEnsembleLoss(teacher),
                                          drop_synthetic_inputs=False)

        student = hydra.utils.instantiate(config.classifier, num_words=vocab_size)
        student = try_cuda(student)

        if config.teacher.ckpt_init.type == 'init':
            assert config.classifier.depth == config.teacher.depth
            assert config.teacher.num_components == 1
            init_teachers, init_fnames = load_teachers(config, ckpt_pattern='*teacher_init_?.ckpt',
                                                       num_words=vocab_size)
            print('initializing the student near the initial teacher weights')
            init_teachers = select_ckpts(init_teachers, config.trial_id, 1, ckpt_names=init_fnames)
            start_idx = (config.trial_id * config.teacher.num_components) % len(init_teachers) - len(init_teachers)
            stop_idx = start_idx + 1
            print(f'using checkpoints {[(len(init_teachers) + i) % len(init_teachers) for i in range(start_idx, stop_idx)]}')
            student = interpolate_net(student, init_teachers[0].state_dict(),
                                      config.teacher.ckpt_init.loc_param, train_loader,
                                      config.trainer.freeze_bn)

        elif config.teacher.ckpt_init.type == 'final':
            assert config.classifier.depth == config.teacher.depth
            assert config.teacher.num_components == 1
            print('initializing the student near the final teacher weights')
            student = interpolate_net(student, teachers[0].state_dict(),
                                      config.teacher.ckpt_init.loc_param, train_loader,
                                      config.trainer.freeze_bn)
            # scale the learning rate down if student is initialized close to teacher
            config.trainer.optimizer.lr = max(
                config.trainer.optimizer.lr * config.teacher.ckpt_init.loc_param,
                config.trainer.lr_scheduler.eta_min
            )
        logger.save_obj(student.state_dict(), 'student_init.ckpt')

        # train_loader, synth_loader = get_distill_loaders(config, train_loader, None)
        student_base_loss = hydra.utils.instantiate(config.loss.init)
        student_loss = distillation.ClassifierStudentLoss(student, student_base_loss, config.loss.alpha)

        print(f"==== distilling student classifier ====")
        student, records = train_loop(
            config,
            student,
            train_closure=distillation_epoch,
            train_loader=distill_loader,
            train_kwargs=dict(loss_fn=student_loss),
            eval_closure=partial(eval_epoch, drop_synthetic_inputs=False, with_cka=False),
            eval_loader=test_loader,
            eval_kwargs=dict(loss_fn=student_loss, teacher=teacher),
            tb_logger=tb_logger,
            tb_prefix="student/",
        )
        for r in records:
            r.update(dict(teacher_train_acc=teacher_train_metrics['test_acc'],
                          teacher_test_acc=teacher_test_metrics['test_acc']))
        logger.add_table(f'student_train_metrics', records)
        logger.write_csv()
        logger.save_obj(student.state_dict(), f'student.ckpt')

        del train_loader, test_loader  # these will be regenerated w/o augmentation
        # save_logits(config, student, teacher, None, logger)

        return 1 - records[-1]['test_acc'] / 100. if len(records) > 0 else float('NaN')

    except Exception:
        logging.error(traceback.format_exc())
        return float('NaN')


if __name__ == '__main__':
    main()

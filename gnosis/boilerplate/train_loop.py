from gnosis.boilerplate import train_epoch, eval_epoch
from hydra.utils import instantiate
from gnosis.utils.metrics import teacher_student_agreement


def train_loop(config, teacher, student, loss_fn, trainloader, testloader, tb_logger, tb_prefix=""):
    optimizer = instantiate(config.trainer.optimizer, params=student.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    records = []
    for epoch in range(config.trainer.num_epochs):
        train_loss, train_acc = train_epoch(student, trainloader, optimizer, loss_fn, lr_scheduler, epoch)
        metrics = dict(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            lr=lr_scheduler.get_last_lr()[0]
        )

        if epoch % config.trainer.eval_period < (config.trainer.eval_period - 1):
            continue

        test_loss, test_acc = eval_epoch(student, testloader, loss_fn)
        train_ts_agree = train_acc
        test_ts_agree = test_acc
        if teacher is not None:
            train_ts_agree = teacher_student_agreement(teacher, student, trainloader)
            test_ts_agree = teacher_student_agreement(teacher, student, testloader)
            print(f'teacher/student train agreement: {train_ts_agree:0.2f}%, '
                  f'test agreement: {test_ts_agree:0.2f}%')
        metrics.update(dict(
            train_ts_agree=train_ts_agree,
            test_loss=test_loss,
            test_acc=test_acc,
            test_ts_agree=test_ts_agree,
        ))
        records.append(metrics)

        tb_logger.add_scalar("{}train/loss".format(tb_prefix), train_loss, epoch)
        tb_logger.add_scalar("{}train/accuracy".format(tb_prefix), train_acc, epoch)
        tb_logger.add_scalar("{}test/accuracy".format(tb_prefix), test_acc, epoch)
        tb_logger.add_scalar("{}test/accuracy".format(tb_prefix), test_acc, epoch)
        tb_logger.add_scalar("{}hypers/learning_rate".format(tb_prefix), lr_scheduler.get_last_lr()[0], epoch)

    return student, records

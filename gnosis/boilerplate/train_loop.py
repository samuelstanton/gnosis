from hydra.utils import instantiate


def train_loop(config, student, train_closure, train_loader, train_kwargs,
               eval_closure, eval_loader, eval_kwargs, tb_logger, tb_prefix=""):
    optimizer = instantiate(config.trainer.optimizer, params=student.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    records = []
    eval_metrics = eval_closure(student, eval_loader, epoch=0, **eval_kwargs)
    records.append(eval_metrics)
    for epoch in range(1, config.trainer.num_epochs + 1):
        metrics = {}
        train_metrics = train_closure(student, train_loader, optimizer,
                                      lr_scheduler, epoch, **train_kwargs)
        metrics.update(train_metrics)

        if epoch % config.trainer.eval_period < (config.trainer.eval_period - 1):
            continue

        eval_metrics = eval_closure(student, eval_loader, epoch=epoch, **eval_kwargs)
        metrics.update(eval_metrics)
        # csv logger
        records.append(metrics)

        # log to tensorboard
        for key, val in train_metrics.items():
            if key == 'epoch':
                continue
            tb_logger.add_scalar(f"{tb_prefix}train/{key}", val, epoch)

        for key, val in eval_metrics.items():
            if key == 'epoch':
                continue
            tb_logger.add_scalar(f"{tb_prefix}eval/{key}", val, epoch)

    return student, records

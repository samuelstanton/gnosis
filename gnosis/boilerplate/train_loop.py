from hydra.utils import instantiate


def train_loop(config, student, train_closure, train_loader, train_kwargs,
               eval_closure, eval_loader, eval_kwargs, tb_logger, tb_prefix=""):
    optimizer = instantiate(config.trainer.optimizer, params=student.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    records = []
    eval_metrics = eval_closure(student, eval_loader, epoch=0, **eval_kwargs)
    records.append(eval_metrics)
    for epoch in range(config.trainer.num_epochs):
        metrics = {}

        train_metrics = train_closure(student, train_loader, optimizer, lr_scheduler,
                                      epoch + 1, config.mixup.alpha, **train_kwargs)
        metrics.update(train_metrics)
        for key, val in train_metrics.items():
            if key == 'epoch':
                continue
            tb_logger.add_scalar(f"{tb_prefix}train/{key}", val, epoch)

        if epoch % config.trainer.eval_period == (config.trainer.eval_period - 1):
            eval_metrics = eval_closure(student, eval_loader, epoch + 1, **eval_kwargs)
            metrics.update(eval_metrics)
            for key, val in eval_metrics.items():
                if key == 'epoch':
                    continue
                tb_logger.add_scalar(f"{tb_prefix}eval/{key}", val, epoch)
            records.append(metrics)

    return student, records

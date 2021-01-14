from hydra.utils import instantiate


def train_loop(config, student, train_closure, train_loader, train_kwargs,
               eval_closure, eval_loader, eval_kwargs, tb_logger, tb_prefix=""):
    optimizer = instantiate(config.trainer.optimizer, params=student.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    records = []
    for epoch in range(config.trainer.num_epochs):
        train_metrics = train_closure(student, train_loader, optimizer,
                                      lr_scheduler, epoch, **train_kwargs)

        if epoch % config.trainer.eval_period < (config.trainer.eval_period - 1):
            continue

        eval_metrics = eval_closure(student, eval_loader, **eval_kwargs)
        train_metrics.update(eval_metrics)
        records.append(train_metrics)

        for key, val in train_metrics.items():
            if key == epoch:
                continue
            tb_logger.add_scalar(f"{tb_prefix}train/{key}", val, epoch)
        for key, val in eval_metrics.items():
            tb_logger.add_scalar(f"{tb_prefix}eval/{key}", val, epoch)

    return student, records

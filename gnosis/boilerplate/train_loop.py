from gnosis.boilerplate import train_epoch, eval_epoch
from hydra.utils import instantiate


def train_loop(config, model, loss_fn, trainloader, testloader, tb_logger, tb_prefix=""):
    optimizer = instantiate(config.trainer.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    records = []
    for epoch in range(config.trainer.num_epochs):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, loss_fn, lr_scheduler, epoch)
        test_loss, test_acc = eval_epoch(model, testloader, loss_fn)

        tb_logger.add_scalar("{}train/loss".format(tb_prefix), train_loss, epoch)
        tb_logger.add_scalar("{}train/accuracy".format(tb_prefix), train_acc, epoch)
        tb_logger.add_scalar("{}test/accuracy".format(tb_prefix), test_acc, epoch)
        tb_logger.add_scalar("{}test/accuracy".format(tb_prefix), test_acc, epoch)
        tb_logger.add_scalar("{}hypers/learning_rate".format(tb_prefix), lr_scheduler.get_last_lr()[0], epoch)

        records.append(dict(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            lr=lr_scheduler.get_last_lr()[0]
        ))

    model.eval()
    return model, records

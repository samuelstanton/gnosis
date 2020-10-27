from torch import nn, optim
from torch.optim import lr_scheduler
from gnosis.boilerplate import train_epoch, eval_epoch
from hydra.utils import instantiate


def train_loop(config, model, trainloader, testloader, s3_logger):
    s3_logger.add_table('train_metrics')

    loss_fn = instantiate(config.trainer.loss_fn)
    optimizer = instantiate(config.trainer.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.trainer.lr_scheduler, optimizer=optimizer)

    for epoch in range(config.trainer.num_epochs):
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, loss_fn, lr_scheduler, epoch)
        test_loss, test_acc = eval_epoch(model, testloader, loss_fn)

        s3_logger.log(dict(
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            lr=lr_scheduler.get_last_lr()[0]
        ), step=epoch + 1, table_name='train_metrics')
        s3_logger.write_csv()

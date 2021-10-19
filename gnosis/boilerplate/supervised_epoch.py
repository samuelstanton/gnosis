from tqdm import tqdm
from upcycle.cuda import try_cuda


def get_lr(lr_scheduler):
    return lr_scheduler.get_last_lr()[0]


def supervised_epoch(net, loader, optimizer, lr_scheduler, epoch, loss_fn):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[LR=%.3f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (get_lr(lr_scheduler), 0, 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = try_cuda(inputs, targets)
        optimizer.zero_grad()
        loss, outputs = loss_fn(inputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%.4f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (get_lr(lr_scheduler), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    lr_scheduler.step()
    metrics = dict(
            train_loss=train_loss / len(loader),
            train_acc=100 * correct / total,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    return metrics

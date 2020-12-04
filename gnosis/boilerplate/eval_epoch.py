import torch
from tqdm import tqdm
from upcycle.cuda import try_cuda


def eval_epoch(net, loader, loss_fn):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = try_cuda(inputs), try_cuda(targets)
            # loss, outputs = loss_fn(inputs, targets)
            loss, outputs, targets = loss_fn(inputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    avg_loss = test_loss / (batch_idx + 1)
    test_acc = 100. * correct / total
    return avg_loss, test_acc
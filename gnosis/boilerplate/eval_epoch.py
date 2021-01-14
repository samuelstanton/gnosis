import torch
from tqdm import tqdm
from upcycle.cuda import try_cuda
from gnosis.distillation.classification import reduce_teacher_logits


def eval_epoch(net, loader, loss_fn, teacher=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    agree = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        with torch.no_grad():
            inputs, targets = try_cuda(inputs, targets)
            loss_args = [inputs, targets]
            if teacher is not None:
                teacher_logits = teacher(inputs).transpose(1, 0)
                loss_args.append(teacher_logits)
            loss, logits = loss_fn(*loss_args)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if teacher is not None:
                teacher_predicted = reduce_teacher_logits(teacher_logits).argmax(-1)
                agree += predicted.eq(teacher_predicted).sum().item()

            desc = ('[eval] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    metrics = dict(
        test_loss=test_loss / len(loader),
        test_acc=100. * correct / total
    )
    if teacher is not None:
        metrics.update(dict(test_ts_agree=100. * agree / total))
    return metrics

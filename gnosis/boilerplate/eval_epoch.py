import torch
from tqdm import tqdm
from upcycle.cuda import try_cuda
from gnosis.distillation.classification import reduce_ensemble_logits
from gnosis.utils.metrics import batch_calibration_stats, expected_calibration_err, ece_bin_metrics


def eval_epoch(net, loader, loss_fn, teacher=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    agree = 0
    ece_stats = None
    desc = ('[eval] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        with torch.no_grad():
            inputs, targets = try_cuda(inputs, targets)
            loss_args = [inputs, targets]
            if teacher is not None:
                teacher_logits = teacher(inputs)
                teacher_logits = reduce_ensemble_logits(teacher_logits)
                loss_args.append(teacher_logits)
            loss, logits = loss_fn(*loss_args)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if teacher is not None:
                teacher_predicted = teacher_logits.argmax(-1)
                agree += predicted.eq(teacher_predicted).sum().item()

            batch_ece_stats = batch_calibration_stats(logits, targets, num_bins=10)
            ece_stats = batch_ece_stats if ece_stats is None else [
                t1 + t2 for t1, t2 in zip(ece_stats, batch_ece_stats)
            ]

            desc = ('[eval] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    ece = expected_calibration_err(*ece_stats, num_samples=total)
    metrics = {
        "test/loss": test_loss / len(loader),
        "test/acc": 100. * correct / total,
        "test/ece": ece
    }

    metrics.update(ece_bin_metrics(*ece_stats, num_bins=10,
                                   prefix='calibration/test'))
    if teacher is not None:
        metrics.update(dict(test_ts_agree=100. * agree / total))
    return metrics

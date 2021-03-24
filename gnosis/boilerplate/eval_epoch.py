import torch
from tqdm import tqdm
from upcycle.cuda import try_cuda
from gnosis.distillation.classification import reduce_ensemble_logits
from gnosis.utils.metrics import batch_calibration_stats, expected_calibration_err, ece_bin_metrics, preact_cka
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F


def eval_epoch(net, loader, epoch, loss_fn, teacher=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    agree = 0
    nll = 0
    kl = 0
    ece_stats = None
    desc = ('[eval] Loss: %.3f | Acc: %.3f (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, batch in prog_bar:
        with torch.no_grad():
            inputs, targets = try_cuda(batch[0], batch[1])
            loss_args = [inputs[:targets.size(0)], targets]  # synthetic data won't be labeled
            if teacher is not None:
                teacher_logits = teacher(inputs)
                teacher_logits = reduce_ensemble_logits(teacher_logits)
                loss_args.append(teacher_logits)
            loss, logits = loss_fn(*loss_args)

            test_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            nll += -F.log_softmax(logits, dim=-1)[..., targets].mean().item()
            if teacher is not None:
                teacher_predicted = teacher_logits.argmax(-1)
                agree += predicted.eq(teacher_predicted).sum().item()
                kl += kl_divergence(
                    Categorical(logits=teacher_logits),
                    Categorical(logits=logits)
                ).mean().item()

            batch_ece_stats = batch_calibration_stats(logits, targets, num_bins=10)
            ece_stats = batch_ece_stats if ece_stats is None else [
                t1 + t2 for t1, t2 in zip(ece_stats, batch_ece_stats)
            ]

            desc = ('[eval] Loss: %.3f | Acc: %.3f (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    ece = expected_calibration_err(*ece_stats, num_samples=total)
    metrics = dict(
        test_loss=test_loss / len(loader),
        test_acc=100. * correct / total,
        test_ece=ece,
        test_nll=nll / len(loader),
        epoch=epoch,
    )
    # metrics.update(ece_bin_metrics(*ece_stats, num_bins=10, prefix='test'))
    if teacher is not None:
        metrics.update(dict(test_ts_agree=100. * agree / total, test_ts_kl=kl / len(loader)))
    if teacher is not None and len(teacher.components) == 1:
        cka = preact_cka(teacher.components[0], net, loader)
        metrics.update({f'test_cka_{i}': val for i, val in enumerate(cka)})

    return metrics

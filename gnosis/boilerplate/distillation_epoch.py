import numpy as np
from tqdm import tqdm
from upcycle.cuda import try_cuda
import torch
from torch.utils.data import DataLoader
from gnosis.distillation.classification import reduce_ensemble_logits
from gnosis.utils.metrics import batch_calibration_stats, expected_calibration_err, ece_bin_metrics


def get_lr(lr_scheduler):
    return lr_scheduler.get_last_lr()[0]


def mixup_data(x, alpha):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = try_cuda(torch.randperm(batch_size))
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x


# def make_generator(teacher, train_loader, synth_loader, mixup_alpha):
#     if synth_loader is None:
#         for input_batch, target_batch in train_loader:
#             input_batch, target_batch = try_cuda(input_batch, target_batch)
#             if mixup_alpha > 0:
#                 input_batch = mixup_data(input_batch, mixup_alpha)
#             with torch.no_grad():
#                 logit_batch = teacher(input_batch)
#                 logit_batch = reduce_ensemble_logits(logit_batch)
#             yield input_batch, target_batch, logit_batch
#     else:
#         for real_batch, synth_batch in zip(train_loader, synth_loader):
#             real_batch = try_cuda(*real_batch)
#             synth_batch = try_cuda(*synth_batch)
#             with torch.no_grad():
#                 real_logits = teacher(try_cuda(real_batch[0]))
#                 real_logits = reduce_ensemble_logits(real_logits)
#                 input_batch = torch.cat([real_batch[0], synth_batch[0]])
#                 target_batch = real_batch[1]
#                 logit_batch = torch.cat([real_logits, synth_batch[2]])
#             yield input_batch, target_batch, logit_batch


def distillation_epoch(student, train_loader, optimizer, lr_scheduler, epoch, mixup_alpha, loss_fn):
    student.train()
    train_loss, correct, agree, total, real_total = 0, 0, 0, 0, 0
    ece_stats = None
    desc = ('[distill] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
            (epoch, get_lr(lr_scheduler), 0, 0, correct, total))
    num_batches = len(train_loader)
    if mixup_alpha > 0 and loss_fn.alpha > 0:
        raise NotImplementedError('Mixup not implemented for hard label distillation loss.')

    prog_bar = tqdm(enumerate(train_loader), total=num_batches, desc=desc, leave=True)
    for batch_idx, (inputs, targets, teacher_logits, temp) in prog_bar:
        optimizer.zero_grad()
        loss, student_logits = loss_fn(inputs, targets, teacher_logits, temp)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        student_predicted = student_logits.argmax(-1)
        teacher_predicted = teacher_logits.argmax(-1)
        full_batch_size = inputs.size(0)
        real_batch_size = targets.size(0)
        total += full_batch_size
        real_total += real_batch_size
        correct += student_predicted[:real_batch_size].eq(targets).sum().item()
        agree += student_predicted.eq(teacher_predicted).sum().item()

        batch_ece_stats = batch_calibration_stats(student_logits[:real_batch_size], targets, num_bins=10)
        ece_stats = batch_ece_stats if ece_stats is None else [
            t1 + t2 for t1, t2 in zip(ece_stats, batch_ece_stats)
        ]

        desc = ('[distill] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
                (epoch, get_lr(lr_scheduler), train_loss / (batch_idx + 1),
                 100. * correct / real_total, correct, real_total))
        prog_bar.set_description(desc, refresh=True)

    lr_scheduler.step()
    ece = expected_calibration_err(*ece_stats, num_samples=total)
    metrics = dict(
            train_loss=train_loss / num_batches,
            train_acc=100 * correct / real_total,
            train_ts_agree=100 * agree / total,
            train_ece=ece,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    metrics.update(ece_bin_metrics(*ece_stats, num_bins=10, prefix='train'))
    return metrics

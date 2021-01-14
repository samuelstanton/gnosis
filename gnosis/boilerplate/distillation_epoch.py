from tqdm import tqdm
from upcycle.cuda import try_cuda
from gnosis.utils.data import get_distill_loader
from gnosis.distillation.classification import reduce_teacher_logits
import torch


def get_lr(lr_scheduler):
    return lr_scheduler.get_last_lr()[0]


def distillation_epoch(student, train_loader, optimizer, lr_scheduler, epoch, loss_fn,
                       teacher, synth_data, config):
    student.train()
    train_loss, correct, agree, total = 0, 0, 0, 0
    desc = ('[student] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
            (epoch, get_lr(lr_scheduler), 0, 0, correct, total))
    with torch.no_grad():
        distill_loader = get_distill_loader(config, teacher, train_loader.dataset, synth_data)
    prog_bar = tqdm(enumerate(distill_loader), total=len(distill_loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets, logits) in prog_bar:
        inputs, targets, logits = try_cuda(inputs, targets, logits)
        optimizer.zero_grad()
        loss, student_logits = loss_fn(inputs, targets, logits)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        student_predicted = student_logits.argmax(-1)
        teacher_predicted = reduce_teacher_logits(logits).argmax(-1)
        total += inputs.size(0)
        correct += student_predicted.eq(targets).sum().item()
        agree += student_predicted.eq(teacher_predicted).sum().item()

        desc = ('[train] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
                (epoch, get_lr(lr_scheduler), train_loss / (batch_idx + 1),
                 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    lr_scheduler.step()
    metrics = dict(
            train_loss=train_loss / len(distill_loader),
            train_acc=100 * correct / total,
            train_ts_agree=100 * agree / total,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    return metrics

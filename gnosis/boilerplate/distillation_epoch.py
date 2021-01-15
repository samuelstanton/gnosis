from tqdm import tqdm
from upcycle.cuda import try_cuda
import torch
from gnosis.distillation.classification import reduce_ensemble_logits


def get_lr(lr_scheduler):
    return lr_scheduler.get_last_lr()[0]


def make_generator(teacher, train_loader, synth_loader):
    if synth_loader is None:
        for input_batch, target_batch in train_loader:
            input_batch, target_batch = try_cuda(input_batch, target_batch)
            with torch.no_grad():
                logit_batch = teacher(input_batch)
                logit_batch = reduce_ensemble_logits(logit_batch)
            yield input_batch, target_batch, logit_batch
    else:
        for real_batch, synth_batch in zip(train_loader, synth_loader):
            real_batch = try_cuda(*real_batch)
            synth_batch = try_cuda(*synth_batch)
            with torch.no_grad():
                real_logits = teacher(try_cuda(real_batch[0]))
                real_logits = reduce_ensemble_logits(real_logits)
                input_batch = torch.cat([real_batch[0], synth_batch[0]])
                target_batch = torch.cat([real_batch[1], synth_batch[1]])
                logit_batch = torch.cat([real_logits, synth_batch[2]])
            yield input_batch, target_batch, logit_batch


def distillation_epoch(student, train_loader, optimizer, lr_scheduler, epoch, loss_fn,
                       teacher, synth_loader):
    student.train()
    train_loss, correct, agree, total = 0, 0, 0, 0
    desc = ('[student] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
            (epoch, get_lr(lr_scheduler), 0, 0, correct, total))
    num_batches = len(train_loader) if synth_loader is None else min(len(train_loader), len(synth_loader))
    batch_generator = make_generator(teacher, train_loader, synth_loader)
    prog_bar = tqdm(enumerate(batch_generator), total=num_batches, desc=desc, leave=True)
    for batch_idx, (inputs, targets, teacher_logits) in prog_bar:
        inputs, targets, teacher_logits = try_cuda(inputs, targets, teacher_logits)
        optimizer.zero_grad()
        loss, student_logits = loss_fn(inputs, targets, teacher_logits)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        student_predicted = student_logits.argmax(-1)
        teacher_predicted = teacher_logits.argmax(-1)
        total += inputs.size(0)
        correct += student_predicted.eq(targets).sum().item()
        agree += student_predicted.eq(teacher_predicted).sum().item()

        desc = ('[train] epoch: %d | lr: %.4f | loss: %.3f | acc: %.3f%% (%d/%d)' %
                (epoch, get_lr(lr_scheduler), train_loss / (batch_idx + 1),
                 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    lr_scheduler.step()
    metrics = dict(
            train_loss=train_loss / num_batches,
            train_acc=100 * correct / total,
            train_ts_agree=100 * agree / total,
            lr=lr_scheduler.get_last_lr()[0],
            epoch=epoch
        )
    return metrics

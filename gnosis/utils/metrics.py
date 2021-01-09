import torch
import torch.nn.functional as F
from upcycle import cuda
import math


def classifier_agreement(logits_1, logits_2):
    preds_1 = torch.argmax(logits_1, dim=-1)
    preds_2 = torch.argmax(logits_2, dim=-1)
    eq_preds = preds_1.eq(preds_2).float()
    return eq_preds.mean().item()


def teacher_student_agreement(teacher, student, dataloader):
    """
    returns the percentage of examples for which the teacher and student
    top-1 predictions agree.
    """
    num_batches = len(dataloader)
    agree_ratio = 0
    for inputs, _ in dataloader:
        inputs = cuda.try_cuda(inputs)
        with torch.no_grad():
            s_logits = student(inputs)
            t_logits = teacher(inputs)
        if t_logits.ndim == 3:
            t_logits = F.log_softmax(t_logits, dim=2)
            t_logits = torch.logsumexp(t_logits, dim=0) - math.log(t_logits.size(0))

        agree_ratio += classifier_agreement(t_logits, s_logits) / num_batches
    return agree_ratio * 100

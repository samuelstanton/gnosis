import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence


class ClassifierTeacherLoss(object):
    def __init__(self, teacher_model):
        self.teacher = teacher_model

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs)
        loss = F.cross_entropy(logits, targets)
        return loss, logits


class ClassifierStudentLoss(object):
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model

    def __call__(self, inputs, *args):
        with torch.no_grad():
            teacher_logp = self.teacher(inputs).log_softmax(dim=-1)

        student_logits = self.student(inputs)
        student_logp = student_logits.log_softmax(dim=-1)

        teacher_dist = Categorical(logits=teacher_logp)
        student_dist = Categorical(logits=student_logp)

        kl_p_q = kl_divergence(teacher_dist, student_dist)
        kl_q_p = kl_divergence(student_dist, teacher_dist)
        loss = kl_p_q.mean() + kl_q_p.mean()

        # loss = F.mse_loss(student_probs, teacher_probs)  # Brier Score

        return loss, student_logits

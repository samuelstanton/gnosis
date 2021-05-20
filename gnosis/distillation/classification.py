import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import kl_divergence
import math
from abc import ABC, abstractmethod


class ClassifierTeacherLoss(object):
    def __init__(self, teacher_model):
        self.teacher = teacher_model

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs)
        loss = F.cross_entropy(logits, targets)
        return loss, logits


class ClassifierStudentLoss(object):
    def __init__(self, student_model, base_loss, alpha=0.9):
        self.student = student_model
        self.base_loss = base_loss
        self.alpha = alpha

    def __call__(self, inputs, targets, teacher_logits, temp=None):
        real_batch_size = targets.size(0)
        student_logits = self.student(inputs)
        hard_loss = F.cross_entropy(student_logits[:real_batch_size], targets)
        # temp = torch.ones_like(student_logits) if temp is None else temp.unsqueeze(-1)
        temp = torch.ones_like(student_logits) if temp is None else temp
        soft_loss = self.base_loss(teacher_logits, student_logits, temp)
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return loss, student_logits


def reduce_ensemble_logits(teacher_logits):
    assert teacher_logits.dim() == 3
    teacher_logits = F.log_softmax(teacher_logits, dim=-1)
    n_teachers = len(teacher_logits)
    return torch.logsumexp(teacher_logits, dim=1) - math.log(n_teachers)


class BaseClassificationDistillationLoss(ABC):
    """Abstract class that defines interface for distillation losses.
    """

    def __call__(self, teacher_logits, student_logits, teacher_temperature=1.):
        """Evaluate loss.

        :param teacher_logits: tensor of teacher model logits of size
            [num_teachers, batch_size, num_classes] or [batch_size, num_classes]
        :param student_logits: tensor of student model logits of size
            [batch_size, num_classes]
        :param teacher_temperature: temperature to apply to the teacher logits
        :return: scalar loss value
        """
        teacher_logits = self._reduce_teacher_predictions(teacher_logits)
        teacher_logits = self._temper_predictions(teacher_logits,
                                                  teacher_temperature)
        assert teacher_logits.shape == student_logits.shape, \
            "Shape mismatch: teacher logits" \
            "have shape {} and student logits have shape {}".format(
                    teacher_logits.shape, student_logits.shape)
        return self.teacher_student_loss(teacher_logits, student_logits)

    @staticmethod
    def _reduce_teacher_predictions(teacher_logits):
        if len(teacher_logits.shape) == 3:
            return reduce_ensemble_logits(teacher_logits)
        return teacher_logits

    @staticmethod
    def _temper_predictions(teacher_logits, teacher_temperature=1.):
        return teacher_logits / teacher_temperature

    @staticmethod
    @abstractmethod
    def teacher_student_loss(teacher_logits, student_logits):
        raise NotImplementedError


class TeacherStudentKLLoss(BaseClassificationDistillationLoss):
    """KL loss between the teacher and student predictions.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits):
        teacher_dist = Categorical(logits=teacher_logits)
        student_dist = Categorical(logits=student_logits)

        return kl_divergence(teacher_dist, student_dist).mean()


class SymmetrizedKLLoss(BaseClassificationDistillationLoss):
    """Symmetrized KL loss.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits):
        teacher_dist = Categorical(logits=teacher_logits)
        student_dist = Categorical(logits=student_logits)

        kl_p_q = kl_divergence(teacher_dist, student_dist)
        kl_q_p = kl_divergence(student_dist, teacher_dist)
        return kl_p_q.mean() + kl_q_p.mean()


class BrierLoss(BaseClassificationDistillationLoss):
    """Brier loss.

    Note: error is averaged both over the classes and data.
    """
    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits):
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        return F.mse_loss(student_probs, teacher_probs)


class AveragedSymmetrizedKLLoss(BaseClassificationDistillationLoss):
    """Symmetrized KL averaged over teacher models.

    Here, instead of using the ensemble, we compute the average KL to each of
    the teacher models. This is the loss that we had implemented originally.
    """

    def __call__(cls, teacher_logits, student_logits, teacher_temperature=1.):
        # overwrite the __call__ method to not reduce the teacher logits
        teacher_logits = cls._temper_predictions(teacher_logits,
                                                 teacher_temperature)
        return cls.teacher_student_loss(teacher_logits, student_logits)

    @staticmethod
    def teacher_student_loss(teacher_logits, student_logits):
        teacher_dist = Categorical(logits=teacher_logits)
        student_dist = Categorical(logits=student_logits)

        kl = kl_divergence(teacher_dist, student_dist).mean()
        reversed_kl = kl_divergence(student_dist, teacher_dist).mean()

        return kl + reversed_kl


class TeacherStudentHardCrossEntLoss(BaseClassificationDistillationLoss):
    """
    Standard cross-entropy loss w.r.t. the hard teacher labels
    """
    def __init__(self, corruption_ratio=0., **kwargs):
        super().__init__()
        self.corruption_ratio = corruption_ratio

    def teacher_student_loss(self, teacher_logits, student_logits):
        batch_size, num_classes = teacher_logits.shape
        teacher_labels = torch.argmax(teacher_logits, dim=-1)
        num_corrupted = int(batch_size * self.corruption_ratio)

        if num_corrupted > 0:
            rand_labels = torch.randint(0, num_classes, (num_corrupted,), device=teacher_labels.device)
            corrupt_idxs = torch.randint(0, batch_size, (num_corrupted,))
            teacher_labels[corrupt_idxs] = rand_labels

        loss = F.cross_entropy(student_logits, teacher_labels)
        return loss


class TeacherStudentFwdCrossEntLoss(object):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    def __init__(self, temp, **kwargs):
        super().__init__()
        self.temp = temp

    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_logp = F.log_softmax(student_logits / temp, dim=-1)
        loss = -(temp ** 2 * teacher_probs * student_logp).sum(-1).mean()
        return loss


class TeacherStudentCvxCrossEnt(object):
    def __init__(self, temp, T_max, beta=0.5):
        self.hard_loss_fn = TeacherStudentHardCrossEntLoss(corruption_ratio=0.)
        self.soft_loss_fn = TeacherStudentFwdCrossEntLoss(temp=temp)
        self._init_beta = beta
        self.beta = beta
        self.t_max = T_max

    def __call__(self, teacher_logits, student_logits, temp):
        hard_loss = self.hard_loss_fn(teacher_logits, student_logits)
        soft_loss = self.soft_loss_fn(teacher_logits, student_logits, temp)
        cvx_loss = self.beta * hard_loss + (1 - self.beta) * soft_loss
        return cvx_loss

    def step(self):
        next_beta = self.beta - self._init_beta / self.t_max
        next_beta = max(next_beta, 0.)
        self.beta = next_beta


class TeacherStudentRevCrossEntLoss(object):
    """Soft teacher/student cross entropy loss from [Hinton et al (2015)]
        (https://arxiv.org/abs/1503.02531)
    """
    def __init__(self, temp, **kwargs):
        super().__init__()
        self.temp = temp

    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)
        teacher_logp = F.log_softmax(teacher_logits / temp, dim=-1)
        student_probs = F.softmax(student_logits / temp, dim=-1)
        loss = -(temp ** 2 * student_probs * teacher_logp).sum(-1).mean()
        return loss

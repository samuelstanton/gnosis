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
    def __init__(
        self, teacher_model, student_model, base_loss, generator_model=None,
        gen_ratio=0.
    ):
        self.teacher = teacher_model
        self.student = student_model
        self.generator = generator_model
        self.gen_ratio = gen_ratio
        self.base_loss = base_loss

    def __call__(self, inputs, *args):
        batch_size = inputs.size(0)
        if self.generator is not None and self.gen_ratio > 0:
            num_generated = math.ceil(batch_size * self.gen_ratio)
            self.generator.eval()
            synth_inputs = self.generator.sample(num_generated)
            inputs = torch.cat([inputs, synth_inputs], dim=0)

        with torch.no_grad():
            teacher_logp = self.teacher(inputs).log_softmax(dim=-1)

        student_logits = self.student(inputs)
        student_logp = student_logits.log_softmax(dim=-1)
        loss = self.base_loss(teacher_logp, student_logp)

        return loss, student_logits[:batch_size]


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
            teacher_logits = F.log_softmax(teacher_logits, dim=-1)
            n_teachers = len(teacher_logits)
            return torch.logsumexp(teacher_logits, dim=0) - math.log(n_teachers)
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


class TeacherStudentCrossEntLoss(BaseClassificationDistillationLoss):
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

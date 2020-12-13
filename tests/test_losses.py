import unittest
import torch
from torch.nn import functional as F

from gnosis import distillation


class TestClassificationDistillationLoss(unittest.TestCase):
    """Unit tests for classification distillation losses.
    """

    @staticmethod
    def _generate_logits_probs(n_teachers=5, n_data=100, n_classes=10):
        teacher_logits = torch.randn(n_teachers, n_data, n_classes)
        student_logits = torch.randn(n_data, n_classes)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_ens_probs = teacher_probs.mean(dim=0)
        return teacher_logits, teacher_ens_probs, student_logits, student_probs

    def test_loss_raises_shape_error(self):
        """Test that giving the loss inconsistent shapes raises runtime errors.
        """
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())

        loss_fn = distillation.TeacherStudentKLLoss()
        exception_type = AssertionError
        with self.assertRaises(exception_type):
            loss_fn(teacher_logits[:, :-1], student_logits)
        with self.assertRaises(exception_type):
            loss_fn(teacher_logits[:, :, :-1], student_logits)
        with self.assertRaises(exception_type):
            loss_fn(teacher_logits, student_logits[:-1])
        with self.assertRaises(exception_type):
            loss_fn(teacher_logits, student_logits[:, :-1])

        # Providing ensemble logits instead of list of teacher logits
        ens_logits = teacher_ens_probs.log()
        with self.assertRaises(exception_type):
            loss_fn(ens_logits[:-1], student_logits)
        with self.assertRaises(exception_type):
            loss_fn(ens_logits[:, :-1], student_logits)
        with self.assertRaises(exception_type):
            loss_fn(ens_logits, student_logits[:-1])
        with self.assertRaises(exception_type):
            loss_fn(ens_logits, student_logits[:, :-1])

    @staticmethod
    def _get_kl(probs_a, probs_b):
        kl = probs_a * (probs_a / probs_b).log()
        kl = kl.sum(dim=1).mean(dim=0)
        return kl

    def test_kl_teacher_student_loss(self):
        """Test TeacherStudentKLLoss.
        """
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())

        kl = self._get_kl(teacher_ens_probs, student_probs)

        loss_fn = distillation.TeacherStudentKLLoss()
        kl_gnosis = loss_fn(teacher_logits, student_logits)
        kl_gnosis_alt = loss_fn(teacher_ens_probs.log(), student_probs.log())
        self.assertAlmostEqual(kl.item(), kl_gnosis.item(), 4)
        self.assertAlmostEqual(kl.item(), kl_gnosis_alt.item(), 4)

    def test_tempering(self):
        """Test tempering the teacher logits.
        """
        T = 0.5
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())
        teacher_ens_probs_t = F.softmax(teacher_ens_probs.log() / T, dim=-1)
        kl = self._get_kl(teacher_ens_probs_t, student_probs)

        loss_fn = distillation.TeacherStudentKLLoss()
        kl_gnosis = loss_fn(teacher_logits, student_logits, T)
        kl_gnosis_alt = loss_fn(teacher_ens_probs.log(), student_probs.log(), T)
        self.assertAlmostEqual(kl.item(), kl_gnosis.item(), 4)
        self.assertAlmostEqual(kl.item(), kl_gnosis_alt.item(), 4)

    def test_symmetrized_kl_loss(self):
        """Test SymmetrizedKLLoss.
        """
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())

        kl = self._get_kl(teacher_ens_probs, student_probs)
        reverse_kl = self._get_kl(student_probs, teacher_ens_probs)
        loss = kl + reverse_kl

        loss_fn = distillation.SymmetrizedKLLoss()
        loss_gnosis = loss_fn(teacher_logits, student_logits)
        loss_gnosis_alt = loss_fn(teacher_ens_probs.log(), student_probs.log())
        self.assertAlmostEqual(loss.item(), loss_gnosis.item(), 4)
        self.assertAlmostEqual(loss.item(), loss_gnosis_alt.item(), 4)

    def test_averaged_symmetrized_kl_loss(self):
        """Test AveragedSymmetrizedKLLoss.
        """
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())

        loss = 0.
        for logits in teacher_logits:
            probs = F.softmax(logits, dim=-1)
            kl = self._get_kl(probs, student_probs)
            reverse_kl = self._get_kl(student_probs, probs)
            loss += kl + reverse_kl

        loss /= len(teacher_logits)

        loss_fn = distillation.AveragedSymmetrizedKLLoss()
        loss_gnosis = loss_fn(teacher_logits, student_logits)
        self.assertAlmostEqual(loss.item(), loss_gnosis.item(), 4)

    def test_brier_loss(self):
        """Test BrierLoss.
        """
        teacher_logits, teacher_ens_probs, student_logits, student_probs = (
            self._generate_logits_probs())
        loss = (teacher_ens_probs - student_probs).pow(2).mean()

        loss_fn = distillation.BrierLoss()
        loss_gnosis = loss_fn(teacher_logits, student_logits)
        loss_gnosis_alt = loss_fn(teacher_ens_probs.log(), student_probs.log())
        self.assertAlmostEqual(loss.item(), loss_gnosis.item(), 4)
        self.assertAlmostEqual(loss.item(), loss_gnosis_alt.item(), 4)


if __name__ == '__main__':
    unittest.main()

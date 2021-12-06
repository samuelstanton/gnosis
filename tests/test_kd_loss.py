from gnosis.distillation.classification import TeacherStudentFwdCrossEntLoss
from gnosis.distillation.classification import reduce_ensemble_logits
import torch


def test_ts_soft_cross_ent():
    batch_size = 2
    num_teachers = 3
    num_classes = 5

    ensemble_logits = torch.randn(batch_size, num_teachers, num_classes)
    teacher_logits = reduce_ensemble_logits(ensemble_logits)
    student_logits = torch.randn(batch_size, num_classes)
    student_logits.requires_grad_(True)

    loss_fn = TeacherStudentFwdCrossEntLoss()
    loss = loss_fn(teacher_logits, student_logits, temp=1.)
    grad_1 = torch.autograd.grad(loss, student_logits)[0]
    print(grad_1)

    teacher_dist = torch.distributions.Categorical(logits=teacher_logits)
    student_dist = torch.distributions.Categorical(logits=student_logits)

    kl_div = torch.distributions.kl.kl_divergence(teacher_dist, student_dist).mean()
    grad_2 = torch.autograd.grad(kl_div, student_logits)[0]
    print(grad_2)

    assert torch.allclose(grad_1, grad_2)

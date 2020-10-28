import torch.nn.functional as F


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
        student_logits = self.student(inputs)
        student_probs = student_logits.softmax(dim=-1)
        teacher_probs = self.teacher(inputs)
        loss = F.mse_loss(student_probs, teacher_probs)
        return loss, student_logits

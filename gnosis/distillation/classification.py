import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import math


class ClassifierTeacherLoss(object):
    def __init__(self, teacher_model):
        self.teacher = teacher_model

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs)
        loss = F.cross_entropy(logits, targets)
        return loss, logits


class ClassifierStudentLoss(object):
    def __init__(self, teacher_model, student_model, generator_model=None, gen_ratio=0.):
        self.teacher = teacher_model
        self.student = student_model
        self.generator = generator_model
        self.gen_ratio = gen_ratio

    def __call__(self, inputs, targets, *args):
        batch_size = inputs.size(0)
        if self.generator is not None and self.gen_ratio > 0:
            num_generated = math.ceil(batch_size * self.gen_ratio)
            self.generator.eval()
            synth_inputs = self.generator.sample(num_generated)
            inputs = torch.cat([inputs, synth_inputs], dim=0)

        with torch.no_grad():
            teacher_logp = self.teacher(inputs).log_softmax(dim=-1)
            
            # temporary
            # print(inputs.shape, teacher_logp.shape)

            teacher_preds = torch.max(torch.logsumexp(teacher_logp, dim=0), dim=-1)[1]
            # print(teacher_logp[0][teacher_preds == 0][:5])
            # print(inputs[teacher_preds == 0][:5])
            
            # print(teacher_preds)
            new_inputs = []
            new_logp = []
            new_targets = []
            teacher_logp = teacher_logp.transpose(0, 1)
            for i in range(10):
                inputs_i = inputs[teacher_preds == i]
                logp_i = teacher_logp[teacher_preds == i]
                y_i = targets[teacher_preds == i]
                
                num_i = len(logp_i)
                perm = torch.randperm(num_i, device=logp_i.device)
                logp_i = logp_i.clone()[perm]
                inputs_i = inputs_i.clone()[perm]
                y_i = y_i.clone()[perm]
                
                new_inputs.append(inputs_i.clone())
                new_logp.append(logp_i.clone())
                new_targets.append(y_i.clone())

            inputs = torch.cat(new_inputs)
            teacher_logp = torch.cat(new_logp)
            targets = torch.cat(new_targets)
            teacher_logp = teacher_logp.transpose(0, 1)
            
            teacher_preds = torch.max(teacher_logp.mean(dim=0), dim=-1)[1]
            # print(teacher_logp[0][teacher_preds==0][:5])
            # print(inputs[teacher_preds == 0][:5])
            # print(teacher_preds)
            # print(inputs.shape, teacher_logp.shape)
            #done temporary

        student_logits = self.student(inputs)
        student_logp = student_logits.log_softmax(dim=-1)

        teacher_dist = Categorical(logits=teacher_logp)
        student_dist = Categorical(logits=student_logp)

        kl_p_q = kl_divergence(teacher_dist, student_dist)
        kl_q_p = kl_divergence(student_dist, teacher_dist)
        loss = kl_p_q.mean() + kl_q_p.mean()

        # loss = F.mse_loss(student_probs, teacher_probs)  # Brier Score

        return loss, student_logits[:batch_size], targets

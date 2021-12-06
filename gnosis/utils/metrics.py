import math
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from scipy.linalg import norm, sqrtm

from upcycle import cuda
from cka.CKA import kernel_CKA


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


def batch_calibration_stats(logits, targets, num_bins):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins).to(logits.device)
    probs, preds = logits.softmax(dim=-1).max(-1)
    bin_correct = torch.zeros(num_bins).float()
    bin_prob = torch.zeros(num_bins).float()
    bin_count = torch.zeros(num_bins).float()
    for idx, conf_level in enumerate(bin_bounds):
        mask = (conf_level - 1 / num_bins < probs) * (probs <= conf_level)
        num_elements = mask.sum().float()
        total_correct = 0. if num_elements < 1 else preds[mask].eq(targets[mask]).sum()
        total_prob = 0. if num_elements < 1 else probs[mask].sum()
        bin_count[idx] = num_elements
        bin_correct[idx] = total_correct
        bin_prob[idx] = total_prob
    return bin_count, bin_correct, bin_prob


def expected_calibration_err(bin_count, bin_correct, bin_prob, num_samples):
    ece = 0
    for count, correct, prob in zip(bin_count, bin_correct, bin_prob):
        if count < 1:
            continue
        ece += count / num_samples * abs(correct / count - prob / count)
    return ece.item()


def ece_bin_metrics(bin_count, bin_correct, bin_prob, num_bins, prefix):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins)
    assert bin_bounds.size(0) == bin_count.size(0)
    bin_acc = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_correct, bin_count))
    bin_conf = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_prob, bin_count))
    metrics = {f"{prefix}_bin_count_{ub:0.2f}": count.item() for ub, count in zip(bin_bounds, bin_count)}
    metrics.update(
        {f"{prefix}_bin_acc_{ub:0.2f}": acc for ub, acc in zip(bin_bounds, bin_acc)}
    )
    metrics.update(
        {f"{prefix}_bin_conf_{ub:0.2f}": conf for ub, conf in zip(bin_bounds, bin_conf)}
    )
    return metrics


def preact_cka(teacher, student, dataloader):
    """
    https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.ipynb
    """
    cka = None
    for inputs, _ in dataloader:
        inputs = cuda.try_cuda(inputs)
        with torch.no_grad():
            teacher_preacts = teacher.preacts(inputs)
            student_preacts = student.preacts(inputs)

        assert len(teacher_preacts) == len(student_preacts)
        batch_cka = np.empty((len(teacher_preacts),))
        for idx, (t_preact, s_preact) in enumerate(zip(teacher_preacts, student_preacts)):
            t_preact = t_preact.cpu().numpy()  # [batch_size x resolution x resolution x channel_size]
            s_preact = s_preact.cpu().numpy()  # [batch_size x resolution x resolution x channel_size]
            avg_t_preact = np.mean(t_preact, axis=(1, 2))
            avg_s_preact = np.mean(s_preact, axis=(1, 2))
            batch_cka[idx] = kernel_CKA(avg_t_preact.T, avg_s_preact.T)

        if cka is None:
            cka = batch_cka / len(dataloader)
        else:
            cka += batch_cka / len(dataloader)

    return cka


# https://github.com/mfinzi/olive-oil-ml/blob/master/oil/utils/metrics.py
class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class Named(type):
    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


class Eval(object):
    def __init__(self, model, on=True):
        self.model = model
        self.on = on

    def __enter__(self):
        self.training_state = self.model.training
        self.model.train(not self.on)

    def __exit__(self, *args):
        self.model.train(self.training_state)


def get_inception():
    """ grabs the pytorch pretrained inception_v3 with resized inputs """
    inception = inception_v3(pretrained=True, transform_input=False)
    upsample = Expression(lambda x: nn.functional.interpolate(x, size=(299, 299), mode='bilinear'))
    model = nn.Sequential(upsample, inception).cuda().eval()
    return model


def get_logits(model, loader):
    """ Extracts logits from a model, dataloader returns a numpy array of size (N, K)
        where K is the number of classes """
    with torch.no_grad(), Eval(model):
        model_logits = lambda mb: model(mb).cpu().data.numpy()
        logits = np.concatenate([model_logits(minibatch) for minibatch in loader], axis=0)
    return logits


def FID_from_logits(logits1, logits2):
    """Computes the FID between logits1 and logits2
        Inputs: [logits1 (N,C)] [logits2 (N,C)] """
    mu1 = np.mean(logits1, axis=0)
    mu2 = np.mean(logits2, axis=0)
    sigma1 = np.cov(logits1, rowvar=False)
    sigma2 = np.cov(logits2, rowvar=False)

    tr = np.trace(sigma1 + sigma2 - 2 * sqrtm(sigma1 @ sigma2))
    distance = norm(mu1 - mu2) ** 2 + tr
    return distance


def IS_from_logits(logits):
    """ Computes the Inception score (IS) from logits of the dataset of size N with C classes.
        Inputs: [logits (N,C)], Outputs: [IS (scalar)]"""
    # E_z[KL(Pyz||Py)] = \mean_z [\sum_y (Pyz log(Pyz) - Pyz log(Py))]
    Pyz = np.exp(logits).transpose()  # Take softmax (up to a normalization constant)
    Pyz /= Pyz.sum(0)[None, :]  # divide by normalization constant
    Py = np.broadcast_to(Pyz.mean(-1)[:, None], Pyz.shape)  # Average over z
    logIS = entropy(Pyz, Py).mean()  # Average over z
    return np.exp(logIS)


cachedLogits = {}


def FID(loader1, loader2):
    """ Computes the Frechet Inception Distance  (FID) between the two image dataloaders
        using pytorch pretrained inception_v3. Requires >2048 imgs for comparison
        Dataloader should be an iterable of minibatched images, assumed to already
        be normalized with mean 0, std 1 (per color)
        """
    model = get_inception()
    logits1 = get_logits(model, loader1)
    if loader2 not in cachedLogits:
        cachedLogits[loader2] = get_logits(model, loader2)
    logits2 = cachedLogits[loader2]
    return FID_from_logits(logits1, logits2)


def IS(loader):
    """Computes the Inception score of a dataloader using pytorch pretrained inception_v3"""
    model = get_inception()
    logits = get_logits(model, loader)
    return IS_from_logits(logits)


def FID_and_IS(loader1, loader2):
    """Computes FID and IS score for loader1 against target loader2 """
    model = get_inception()
    logits1 = get_logits(model, loader1)
    if loader2 not in cachedLogits:
        cachedLogits[loader2] = get_logits(model, loader2)
    logits2 = cachedLogits[loader2]
    return FID_from_logits(logits1, logits2), IS_from_logits(logits1)

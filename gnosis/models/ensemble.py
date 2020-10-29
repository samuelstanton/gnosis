import torch
import torch.nn.functional as F


class ClassifierEnsemble(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.components = torch.nn.ModuleList(models)

    def forward(self, inputs):
        return torch.stack([model(inputs) for model in self.components])


class ClassifierEnsembleLoss(object):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __call__(self, inputs, targets):
        logits = self.ensemble(inputs)
        log_p = logits.softmax(dim=-1).mean(0).log()
        return F.nll_loss(log_p, targets), log_p

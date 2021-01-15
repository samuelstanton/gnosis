import torch
import torch.nn.functional as F
from gnosis.distillation.classification import reduce_ensemble_logits


class ClassifierEnsemble(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.components = torch.nn.ModuleList(models)

    def forward(self, inputs):
        """[batch_size x num_components x ...]"""
        return torch.stack([model(inputs) for model in self.components], dim=1)


class ClassifierEnsembleLoss(object):
    def __init__(self, ensemble):
        self.ensemble = ensemble

    def __call__(self, inputs, targets):
        logits = self.ensemble(inputs)
        logits = reduce_ensemble_logits(logits)
        return F.nll_loss(logits, targets), logits

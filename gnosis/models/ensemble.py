import torch


class ClassifierEnsemble(torch.nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.components = torch.nn.ModuleList(models)

    def forward(self, inputs):
        logits = torch.stack([model(inputs) for model in self.components])
        probs = logits.softmax(dim=-1)
        return probs.mean(0)

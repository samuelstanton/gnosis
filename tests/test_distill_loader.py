import torch
from torch.utils.data import TensorDataset
from gnosis.distillation.dataloaders import DistillLoader


def test_distillation_loader():
    num_total = 64
    batch_size = 16

    datasets = [
        TensorDataset(torch.rand(num_total // 4, 2, 2), torch.rand(num_total // 4)),
        TensorDataset(torch.rand(3 * (num_total // 4), 2, 2), torch.rand(3 * num_total // 4))
    ]
    loader = DistillLoader(
        teacher=lambda x: x,
        datasets=datasets,
        splits=[0],
        mixup_alpha=0.,
        mixup_portion=1.,
        synth_ratio=0.,
        temp=[1.0, 2.0],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    for inputs, targets, logits, temp in loader.generator:
        assert inputs.size(0) == batch_size
        assert targets.size(0) == batch_size
        assert logits.size(0) == batch_size
        assert temp.size(0) == batch_size
        assert torch.all(temp[:batch_size // 4] == 1.0)
        assert torch.all(temp[batch_size // 4:] == 2.0)

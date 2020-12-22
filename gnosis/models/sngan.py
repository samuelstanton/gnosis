import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import DictConfig
import hydra


class SNGAN(nn.Module):
    def __init__(self, gen_cfg: DictConfig, disc_cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.generator = hydra.utils.instantiate(gen_cfg)
        self.discriminator = hydra.utils.instantiate(disc_cfg)

    def gen_backward(self, batch_size):
        # Generator hinge loss
        fake_samples = self.generator.sample(batch_size)
        fake_logits = self.discriminator(fake_samples)
        loss = -torch.mean(fake_logits)
        loss.backward()
        return loss, fake_samples

    def disc_backward(self, real_samples):
        # Discriminator hinge loss
        batch_size = real_samples.size(0)
        with torch.no_grad():
            fake_samples = self.generator.sample(batch_size)
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        loss.backward()
        return loss

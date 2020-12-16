import torch
from upcycle import cuda
from tqdm import tqdm


def dcgan_epoch(gen_net, disc_net, gen_opt, disc_opt, loss_fn, dataloader):
    real_label, fake_label = 1., 0.
    num_batches = len(dataloader)
    tot_gen_loss = tot_disc_loss = 0
    tot_real_prob = tot_fake_prob = 0
    desc = f'[DC-GAN] D-LOSS: {tot_disc_loss:0.4f}, G-LOSS: {tot_gen_loss: 0.4f}, ' \
           f'R-PROB: {tot_real_prob:0.4f}, G-PROB: {tot_fake_prob:0.4f}'
    prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=True)
    for i, (inputs, _) in prog_bar:
        inputs = cuda.try_cuda(inputs)
        batch_size = inputs.size(0)

        # generate fake image batch
        noise = torch.randn(batch_size, gen_net.input_dim, 1, 1).to(inputs.device)
        fake = gen_net(noise)

        # backprop discriminator grads on real examples
        disc_net.zero_grad()
        disc_real_outputs = disc_net(inputs).view(-1)
        labels = torch.full((batch_size,), real_label, device=inputs.device)
        disc_real_loss = loss_fn(disc_real_outputs, labels)
        disc_real_loss.backward()

        # backprop discriminator grads on fake examples
        disc_fake_outputs = disc_net(fake.detach()).view(-1)
        labels.fill_(fake_label)
        disc_fake_loss = loss_fn(disc_fake_outputs, labels)
        disc_fake_loss.backward()
        # update discriminator
        disc_opt.step()

        # backprop generator grads
        gen_net.zero_grad()
        disc_fake_outputs = disc_net(fake).view(-1)
        labels.fill_(real_label)
        gen_loss = loss_fn(disc_fake_outputs, labels)
        gen_loss.backward()
        # update generator
        gen_opt.step()

        # logging
        tot_gen_loss += gen_loss.item()
        tot_disc_loss += (disc_real_loss + disc_fake_loss).item()
        tot_real_prob += disc_real_outputs.mean(0).item()
        tot_fake_prob += disc_fake_outputs.mean(0).item()
        desc = f'[DC-GAN] D-LOSS: {tot_disc_loss / (i + 1):0.4f}, '\
               f'G-LOSS: {tot_gen_loss / (i + 1): 0.4f}, ' \
               f'R-PROB: {tot_real_prob / (i + 1):0.4f}, G-PROB: {tot_fake_prob / (i + 1):0.4f}'
        prog_bar.set_description(desc, refresh=True)

    metrics = dict(gen_loss=tot_gen_loss / num_batches, disc_loss=tot_disc_loss / num_batches,
                   disc_real_prob=tot_real_prob / num_batches, disc_fake_prob=tot_fake_prob / num_batches)

    return metrics

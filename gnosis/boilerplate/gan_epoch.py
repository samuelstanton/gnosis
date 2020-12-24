from upcycle import cuda
from tqdm import tqdm
from oil.utils import metrics as oil_metrics
from oil.utils.utils import dmap, LoaderTo
from oil.model_trainers.gan import GanLoader
from torch.utils.data import DataLoader


def gan_train_epoch(gan_module, gen_opt, disc_opt, gen_lr_sched, disc_lr_sched,
                    train_loader, test_loader, train_cfg):
    num_batches = len(train_loader)
    gan_module.train()
    gen_update_count = 0
    while gen_update_count < train_cfg.eval_period:
        tot_gen_loss = tot_disc_loss = 0
        desc = f'[GAN] D-LOSS: {tot_disc_loss:0.4f}, G-LOSS: {tot_gen_loss: 0.4f}'
        prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=desc, leave=True)
        for i, (real_samples, _) in prog_bar:
            real_samples = cuda.try_cuda(real_samples)
            batch_size = real_samples.size(0)

            if i % train_cfg.gen_update_period == 0:
                gen_opt.zero_grad()
                gen_loss, _ = gan_module.gen_backward(batch_size)
                gen_opt.step()
                gen_update_count += 1

            if i % train_cfg.disc_update_period == 0:
                disc_opt.zero_grad()
                disc_loss = gan_module.disc_backward(real_samples)
                disc_opt.step()

            # step lr here, assuming decay is pegged to gen_update_count
            if i % train_cfg.gen_update_period == 0:
                gen_lr_sched.step()
                disc_lr_sched.step()

            # logging
            tot_gen_loss += gen_loss.item()
            tot_disc_loss += disc_loss.item()

            desc = f'[GAN] D-LOSS: {tot_disc_loss / (i + 1):0.4f}, '\
                   f'G-LOSS: {tot_gen_loss / (i + 1): 0.4f}'
            prog_bar.set_description(desc, refresh=True)

            if gen_update_count == train_cfg.eval_period:
                break
    gan_module.eval()

    metrics = dict(gen_loss=tot_gen_loss / num_batches, disc_loss=tot_disc_loss / num_batches)
    # get FID and IS
    gan_loader = GanLoader(gan_module.generator, N=train_cfg.num_eval_samples, bs=batch_size)
    test_image_loader = DataLoader(dmap(lambda mb: mb[0], test_loader.dataset), batch_size=batch_size)
    test_image_loader = LoaderTo(test_image_loader, device=gan_module.generator.device)
    fid_score, is_score = oil_metrics.FID_and_IS(gan_loader, test_image_loader)
    metrics.update(dict(fid_score=float(fid_score), is_score=float(is_score)))

    return metrics

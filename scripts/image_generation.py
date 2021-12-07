import hydra
from torch import optim
from upcycle.scripting import startup
from gnosis.utils.data import get_loaders
from gnosis.boilerplate import gan_train_epoch
from upcycle import cuda
from gnosis.utils.optim import get_decay_fn
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf


@hydra.main(config_path='../config/', config_name='image_generation.yaml')
def main(config):
    config, logger = startup(config)
    tb_logger = SummaryWriter(log_dir=f"./{config.job_name}/{config.timestamp}")
    tb_logger.add_text("hypers/transforms", ''.join(config.augmentation.transforms_list), 0)
    tb_logger.add_text("hypers/hydra_cfg", OmegaConf.to_yaml(config))

    train_loader, test_loader, train_splits = get_loaders(config)

    gan = hydra.utils.instantiate(config.density_model)
    gan = cuda.try_cuda(gan)

    opt_betas = (config.trainer.optimizer.beta1, config.trainer.optimizer.beta2)
    gen_opt = optim.Adam(gan.generator.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)
    disc_opt = optim.Adam(gan.discriminator.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)
    # linear LR decay
    decay_fn = get_decay_fn(config.trainer.optimizer.lr, config.trainer.lr_decay.min_lr,
                            config.trainer.lr_decay.start, config.trainer.lr_decay.stop)
    gen_lr_sched = optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda=decay_fn)
    disc_lr_sched = optim.lr_scheduler.LambdaLR(disc_opt, lr_lambda=decay_fn)

    logger.add_table('train_metrics')
    gen_update_count = 0
    assert config.trainer.eval_period % config.trainer.checkpoint_period == 0
    while gen_update_count < config.trainer.num_gen_updates:
        metrics = gan_train_epoch(gan, gen_opt, disc_opt, gen_lr_sched, disc_lr_sched,
                                  train_loader, test_loader, config.trainer)
        gen_update_count += metrics['num_gen_updates']
        last_lr = gen_lr_sched.get_last_lr()[0]

        print(f'[GAN] : step {gen_update_count}, lr: {last_lr:.6f}, ' \
              f'FID: {metrics["fid_score"]:.4f}, IS: {metrics["is_score"]:.4f}')
        logger.log(metrics, gen_update_count, 'train_metrics')
        logger.write_csv()
        if gen_update_count % config.trainer.checkpoint_period == 0:
            logger.save_obj(gan.generator.state_dict(),
                            f'generator_{gen_update_count}.ckpt')
            logger.save_obj(gan.discriminator.state_dict(),
                            f'discriminator_{gen_update_count}.ckpt')


if __name__ == '__main__':
    main()

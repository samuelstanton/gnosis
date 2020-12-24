import hydra
from torch import optim
from gnosis.utils.scripting import startup
from gnosis.utils.data import get_loaders
from gnosis.boilerplate import gan_train_epoch
from upcycle import cuda
from gnosis.utils.optim import get_decay_fn


@hydra.main(config_path='../config/', config_name='image_generation.yaml')
def main(config):
    config, logger, _ = startup(config)
    trainloader, testloader = get_loaders(config)

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
    while gen_update_count < config.trainer.num_gen_updates:
        metrics = gan_train_epoch(gan, gen_opt, disc_opt, gen_lr_sched, disc_lr_sched,
                                  trainloader, testloader, config.trainer)
        gen_update_count += config.trainer.eval_period
        last_lr = gen_lr_sched.get_last_lr()[0]

        print(f'[GAN] : step {gen_update_count}, lr: {last_lr:.6f}, ' \
              f'FID: {metrics["fid_score"]:.4f}, IS: {metrics["is_score"]:.4f}')
        logger.log(metrics, gen_update_count, 'train_metrics')
        logger.write_csv()
        if gen_update_count == config.trainer.checkpoint_period:
            logger.save_obj(gan.generator.state_dict(),
                            f'{config.dataset.name}_generator_{gen_update_count}.ckpt')
            logger.save_obj(gan.discriminator.state_dict(),
                            f'{config.dataset.name}_discriminator_{gen_update_count}.ckpt')


if __name__ == '__main__':
    main()

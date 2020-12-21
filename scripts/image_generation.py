import hydra
from torch import optim
from gnosis.utils.scripting import startup
from gnosis.utils.data import get_loaders
from gnosis.boilerplate import gan_train_epoch
from upcycle import cuda


@hydra.main(config_path='../config/', config_name='image_generation.yaml')
def main(config):
    config, logger, _ = startup(config)
    trainloader, testloader = get_loaders(config)

    gan = hydra.utils.instantiate(config.density_model)
    gan = cuda.try_cuda(gan)

    opt_betas = (config.trainer.optimizer.beta1, config.trainer.optimizer.beta2)
    gen_opt = optim.Adam(gan.generator.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)
    disc_opt = optim.Adam(gan.discriminator.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)

    logger.add_table('train_metrics')
    for epoch in range(config.trainer.num_epochs):
        metrics = gan_train_epoch(gan, gen_opt, disc_opt, trainloader, testloader, config.trainer)
        print(f'[GAN] epoch: {epoch + 1}, FID: {metrics["fid_score"]:0.4f}, IS: {metrics["is_score"]:0.4f}')
        logger.log(metrics, epoch + 1, 'train_metrics')
        logger.write_csv()
        if epoch % config.checkpoint_freq == (config.checkpoint_freq - 1):
            logger.save_obj(gan.generator.state_dict(), f'{config.dataset.name}_generator_{epoch + 1}.ckpt')
            logger.save_obj(gan.discriminator.state_dict(), f'{config.dataset.name}_discriminator_{epoch + 1}.ckpt')


if __name__ == '__main__':
    main()

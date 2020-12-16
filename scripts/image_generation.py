import hydra
from torch import optim, nn
from gnosis.utils.scripting import startup
from gnosis.utils.data import get_loaders
from upcycle import cuda
from gnosis.boilerplate import dcgan_epoch


@hydra.main(config_path='../config/', config_name='image_generation.yaml')
def main(config):
    config, logger, _ = startup(config)
    trainloader, _ = get_loaders(config)

    gen_net = hydra.utils.instantiate(config.generator)
    disc_net = hydra.utils.instantiate(config.discriminator)
    gen_net, disc_net = cuda.try_cuda(gen_net, disc_net)

    opt_betas = (config.trainer.optimizer.beta1, config.trainer.optimizer.beta2)
    gen_opt = optim.Adam(gen_net.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)
    disc_opt = optim.Adam(disc_net.parameters(), lr=config.trainer.optimizer.lr, betas=opt_betas)
    loss_fn = nn.BCELoss()

    logger.add_table('train_metrics')
    for epoch in range(config.trainer.num_epochs):
        print(f'---- EPOCH {epoch + 1} ----')
        metrics = dcgan_epoch(gen_net, disc_net, gen_opt, disc_opt, loss_fn, trainloader)
        logger.log(metrics, epoch + 1, 'train_metrics')
        logger.write_csv()
        if epoch % config.checkpoint_freq == (config.checkpoint_freq - 1):
            logger.save_obj(gen_net.state_dict(), f'{config.dataset.name}_generator_{epoch + 1}.ckpt')
            logger.save_obj(disc_net.state_dict(), f'{config.dataset.name}_discriminator_{epoch + 1}.ckpt')


if __name__ == '__main__':
    main()

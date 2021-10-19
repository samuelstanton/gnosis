from collections import OrderedDict

import torch
from upcycle import cuda


def interpolate_net(net, ckpt_state_dict, distance_ratio, dataloader, freeze_bn=False):
    interp_state_dict = OrderedDict()
    for (net_key, net_param), (ckpt_key, ckpt_param) in zip(net.state_dict().items(), ckpt_state_dict.items()):
        interp_state_dict[net_key] = (
            distance_ratio * net_param + (1 - distance_ratio) * ckpt_param
        )
    net.load_state_dict(interp_state_dict)
    if freeze_bn:
        return net

    # update batchnorm running stats
    net.train()
    for inputs, _ in dataloader:
        with torch.no_grad():
            net(cuda.try_cuda(inputs))
    net.eval()
    return net

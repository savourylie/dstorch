import torch.nn as nn


def random_weight_init(module, itype='kaiming'):
    for layer in module.modules():
        classname = layer.__class__.__name__
        if classname.find('Conv') != -1:
            layer.weight.data = nn.init.kaiming_normal_(layer.weight.data)
        elif classname.find('BatchNorm') != -1:
            layer.weight.data.normal_(1.0, 0.02)
            layer.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            layer.weight.data = nn.init.kaiming_normal_(layer.weight.data)

    return module
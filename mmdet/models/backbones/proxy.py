import torch.nn as nn
import math

from .ProxylessNAS.proxyless_nas import proxyless_cpu, proxyless_gpu, proxyless_mobile, proxyless_mobile_14

from ..registry import BACKBONES

@BACKBONES.register_module
class Proxy(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(Proxy, self).__init__()
        self.t = proxyless_mobile(pretrained=True) # Yes, we provide pre-trained models!
    def forward(self, x):
        x = self.t.first_conv(x)
        out = []
        for i, l in enumerate(self.t.blocks):
            x = l(x)
            if i in [16, 20, 21]:
                out.append(x)
        x = self.t.feature_mix_layer(x)
        out.append(x)
        return tuple(out)

    def init_weights(self, pretrained=False):
        pass



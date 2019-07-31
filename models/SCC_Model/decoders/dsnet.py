import torch
from torch import nn
from ..layer_factory import convbnrelu


class DDCB(nn.Module):
    def __init__(self, inp, rates):
        super(DDCB, self).__init__()
        self.layers = nn.ModuleList()
        mip = inp // 2
        oup = mip // 2

        planes = inp
        for rate in rates:
            self.layers.append(nn.Sequential(
                convbnrelu(planes, mip, 1),
                convbnrelu(mip, oup, 3, dilation=rate),
            ))
            planes = planes + oup
        self.last_conv = convbnrelu(planes, inp, 3)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            x = torch.cat(features, dim=1)
        return self.last_conv(x)


class DSNet(nn.Module):
    def __init__(self, inp, num_classes,
                 num_blocks=3, rates=(1, 2, 3),
                 dense_skip=False):
        super(DSNet, self).__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            self.blocks.append(DDCB(inp, rates))
        self.last_block = nn.Sequential(
            convbnrelu(inp, 128, 3),
            convbnrelu(128, 64, 3),
            convbnrelu(64, 1, 1))
        self.dense_skip = dense_skip

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            x = block(x) + x
            if self.dense_skip:
                for s in features[:-1]:
                    x = x + s
                features.append(x)
        return self.last_block(x)

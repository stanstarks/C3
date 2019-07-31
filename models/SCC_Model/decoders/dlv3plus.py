import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layer_factory import CRPBlock, batchnorm, conv1x1, conv3x3, sepconv_bn
from ...misc.utils import make_list


class DLv3plus(nn.Module):
    """DeepLab-v3+ for Semantic Image Segmentation.

    ASPP with decoder. Allows to have multiple skip-connections.
    More information about the model: https://arxiv.org/abs/1802.02611

    Args:
      input_sizes (int, or list): number of channels for each input.
                                  Last value represents the input to ASPP,
                                  other values are for skip-connections.
      num_classes (int): number of output channels.
      skip_size (int): common filter size for skip-connections.
      agg_size (int): common filter size.
      rates (list of ints): dilation rates in the ASPP module.

    """
    def __init__(
            self,
            input_sizes,
            num_classes,
            skip_size=48,
            agg_size=256,
            rates=(6, 12, 18)):
        super(DLv3plus, self).__init__()

        skip_convs = nn.ModuleList()
        aspp = nn.ModuleList()

        input_sizes = make_list(input_sizes)

        for size in input_sizes[:-1]:
            skip_convs.append(
                nn.Sequential(
                    conv1x1(size, skip_size, bias=False),
                    batchnorm(skip_size),
                    nn.ReLU(inplace=False)))
        # ASPP
        aspp.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                conv1x1(input_sizes[-1], agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False)))
        aspp.append(
            nn.Sequential(
                conv1x1(input_sizes[-1], agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False)))
        for rate in rates:
            aspp.append(
                sepconv_bn(
                    input_sizes[-1],
                    agg_size,
                    rate=rate,
                    depth_activation=True))
        aspp.append(
            nn.Sequential(
                conv1x1(agg_size * 5, agg_size, bias=False),
                batchnorm(agg_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=0.1)))

        self.skip_convs = skip_convs
        self.aspp = aspp
        self.dec = nn.Sequential(
            sepconv_bn(
                agg_size + len(skip_convs) * skip_size,
                agg_size,
                depth_activation=True),
            sepconv_bn(
                agg_size,
                agg_size,
                depth_activation=True))
        self.clf = conv1x1(agg_size, num_classes, bias=True)

    def forward(self, xs):
        xs = make_list(xs)
        skips = [conv(x) for conv, x in zip(self.skip_convs, xs[:-1])]
        aspp = [branch(xs[-1]) for branch in self.aspp[:-1]]
        # Upsample GAP
        aspp[0] = F.interpolate(
            aspp[0],
            size=xs[-1].size()[2:],
            mode='bilinear',
            align_corners=True)
        aspp = torch.cat(aspp, dim=1)
        # Apply last conv in ASPP
        aspp = self.aspp[-1](aspp)
        # Connect with skip-connections
        dec = [skips[0]]
        for x in skips[1:] + [aspp]:
            dec.append(
                F.interpolate(
                    x,
                    size=dec[0].size()[2:],
                    mode='bilinear',
                    align_corners=True))
        dec = torch.cat(dec, dim=1)
        dec = self.dec(dec)
        out = self.clf(dec)
        return out

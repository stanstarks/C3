from torch import nn
from torch.nn import functional as F

from .mobilenetv2 import mobilenetv2
from .decoders import DLv3plus


class Mnv2Dlv3p(nn.Module):
    """
    MobileNetv2 encoder with DeepLab V3+ decoder
    """
    def __init__(self, pretrained=True):
        super(Mnv2Dlv3p, self).__init__()
        enc = mobilenetv2(
            pretrained=pretrained, return_idx=[2, 6],
            rates=(1, 2, 3))
        dec = DLv3plus(enc._out_c, num_classes=1, rates=enc.rates)
        self.net = nn.Sequential(enc, dec)

    def forward(self, x):
        x = self.net(x)
        x = F.upsample(
            x, scale_factor=8, mode='bilinear', align_corners=False)
        return F.relu(x)

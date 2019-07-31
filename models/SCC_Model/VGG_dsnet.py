import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .decoders.dsnet import DSNet

# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class VGGDSNet(nn.Module):
    def __init__(self, pretrained=True):
        super(VGGDSNet, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

        self.decoder = DSNet(512, 1, dense_skip=False)

    def forward(self, x):
        x = self.features4(x)
        x = self.decoder(x)
        x = F.relu(F.upsample(x, scale_factor=8))

        return x

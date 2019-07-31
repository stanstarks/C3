import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net
        elif model_name == 'mnv2_dlv3p':
            from .SCC_Model.mnv2_dlv3p import Mnv2Dlv3p as net
        elif model_name == 'VGG_DSNet':
            from .SCC_Model.VGG_dsnet import VGGDSNet as net

        self.CCN = net()
        if len(gpus) > 1:
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).cuda()
        else:
            self.CCN = self.CCN.cuda()
        self.mse = nn.MSELoss().cuda()
        self.cns_weight = 1.0

    def forward(self, img, gt_map):
        density_map = self.CCN(img)
        loss_mse = self.mse(density_map.squeeze(), gt_map.squeeze())
        loss_cns = self.consistency_loss(density_map, gt_map) * self.cns_weight
        losses = {"mse": loss_mse, "cns": loss_cns, "total_loss": loss_cns + loss_mse}
        return density_map, losses

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map

    def consistency_loss(self, density_map, gt_data, out_sizes=(1, 2, 4)):
        loss = 0
        gt_data = gt_data.unsqueeze(1)
        for out_size in out_sizes:
            pred = F.adaptive_avg_pool2d(density_map, out_size)
            target = F.adaptive_avg_pool2d(gt_data, out_size)
            l1_dist = F.l1_loss(pred, target)
            loss = loss + l1_dist
        return loss


import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (DisparityWarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, DisparityEstimator, DisparityContextNetwork)
# from correlation_package.modules.correlation import Correlation


class PWCDNet(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level):
        super(PWCDNet, self).__init__()
        self.device = device
        self.lv_chs = lv_chs
        self.num_levels = len(lv_chs)
        self.search_range = search_range
        self.batch_norm = batch_norm
        self.corr_activation = corr_activation
        self.residual = residual
        self.output_level = output_level

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs, batch_norm).to(device)
        
        self.warping_layer = DisparityWarpingLayer(device)

        # if args.corr == 'CostVolumeLayer':
        #     self.corr = CostVolumeLayer(device, search_range)
        # else:
        #     self.corr = Correlation(pad_size = search_range, kernel_size = 1, max_displacement = search_range, stride1 = 1, stride2 = 1, corr_multiply = 1).to(device)

        self.corr = CostVolumeLayer(device, search_range)
        
        self.disparity_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
            layer = DisparityEstimator(ch + (search_range*2+1)**2 + 1, batch_norm).to(device)
            self.add_module(f'DisparityEstimator(Lv{l})', layer)
            self.disparity_estimators.append(layer)
        
        self.context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityContextNetwork(ch + 2, batch_norm).to(device)
            layer = DisparityContextNetwork(ch + 1, batch_norm).to(device)
            self.add_module(f'ContextNetwork(Lv{l})', layer)
            self.context_networks.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw):

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # outputs
        # flows = []

        # tensors for summary
        summaries = {
            'x2_warps': [],

        }

        # for x1 in x1_pyramid:
            # print(x1.shape)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 1
                disp = torch.zeros(shape).to(self.device)
            else:
                output_spatial_size = [x2.size(2), x2.size(3)]
                # flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                disp = F.interpolate(disp, output_spatial_size, mode='bilinear', align_corners=True) * 2 

            x2_warp = self.warping_layer(x2, disp)
            
            # correlation
            corr = self.corr(x1, x2_warp)
            if self.corr_activation: F.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                disp_coarse = self.disparity_estimators[l](torch.cat([x1, corr, disp], dim = 1)) + disp
            else:
                disp_coarse = self.disparity_estimators[l](torch.cat([x1, corr, disp], dim = 1))

            # print(disp_coarse.size())

            disp_fine = self.context_networks[l](torch.cat([x1, disp], dim = 1))
            disp = disp_coarse + disp_fine


            if l == self.output_level:
                disp = F.interpolate(disp, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                # flows.append(flow)
                summaries['x2_warps'].append(x2_warp.data)
                break
            else:
                # flows.append(flow)
                summaries['x2_warps'].append(x2_warp.data)

        return disp

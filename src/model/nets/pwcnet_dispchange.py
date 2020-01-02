import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (DisparityWarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, DisparityEstimator, DisparityContextNetwork)
# from correlation_package.modules.correlation import Correlation


class PWCDCNet(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level):
        super(PWCDCNet, self).__init__()
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

        self.dispchange_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
            layer = DisparityEstimator(ch + ((search_range*2+1)**2)*2 + 1*2, batch_norm).to(device)
            self.add_module(f'DispchangeEstimator(Lv{l})', layer)
            self.dispchange_estimators.append(layer)
        
        self.context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityContextNetwork(ch + 2, batch_norm).to(device)
            layer = DisparityContextNetwork(ch + 1, batch_norm).to(device)
            self.add_module(f'ContextNetwork(Lv{l})', layer)
            self.context_networks.append(layer)

        self.dispchange_context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityContextNetwork(ch + 2, batch_norm).to(device)
            layer = DisparityContextNetwork(ch*2 + 1, batch_norm).to(device)
            self.add_module(f'DispchangeContextNetwork(Lv{l})', layer)
            self.dispchange_context_networks.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw, x1_next_raw, x2_next_raw):

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        x1_next_pyramid = self.feature_pyramid_extractor(x1_next_raw) + [x1_next_raw]
        x2_next_pyramid = self.feature_pyramid_extractor(x2_next_raw) + [x2_next_raw]


        for l, (x1, x2, x1_next, x2_next) in enumerate(zip(x1_pyramid, x2_pyramid, x1_next_pyramid, x2_next_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 1
                dispchange = torch.zeros(shape).to(self.device)
            else:
                output_spatial_size = [x2.size(2), x2.size(3)]
                # flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                dispchange = F.interpolate(dispchange, output_spatial_size, mode='bilinear', align_corners=True) * 2

            x2_warp = self.warping_layer(x2, disp)
            x2_next_warp = self.warping_layer(x2_next, disp_next)
            
            # correlation
            corr = self.corr(x1, x2_warp)
            if self.corr_activation: F.leaky_relu_(corr)
            corr_next = self.corr(x1_next, x2_next_warp)
            if self.corr_activation: F.leaky_relu_(corr_next)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                disp_coarse = self.disparity_estimators[l](torch.cat([x1, corr, disp], dim = 1)) + disp
                disp_next_coarse = self.disparity_estimators[l](torch.cat([x1_next, corr_next, disp_next], dim = 1)) + disp_next
                dispchange_coarse = self.dispchange_estimators[l](torch.cat([x1, disp, disp_next, corr, corr_next], dim = 1)) + dispchange
            else:
                disp_coarse = self.disparity_estimators[l](torch.cat([x1, corr, disp], dim = 1))
                disp_next_coarse = self.disparity_estimators[l](torch.cat([x1_next, corr_next, disp_next], dim = 1))
                dispchange_coarse = self.dispchange_estimators[l](torch.cat([x1, disp, disp_next, corr, corr_next], dim = 1))

            # print(disp_coarse.size())

            disp_fine = self.context_networks[l](torch.cat([x1, disp], dim = 1))
            disp_next_fine = self.context_networks[l](torch.cat([x1_next, disp_next], dim = 1))
            dispchange_fine = self.dispchange_context_networks[l](torch.cat([x1, x1_next, dispchange], dim = 1))

            disp = disp_coarse + disp_fine
            disp_next = disp_next_coarse + disp_next_fine
            dispchange = dispchange_coarse + dispchange_fine

            if l == self.output_level:
                dispchange = F.interpolate(dispchange, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                break

        return dispchange
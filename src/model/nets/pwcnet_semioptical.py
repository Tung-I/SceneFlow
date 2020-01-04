import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
# from correlation_package.modules.correlation import Correlation


class PWCSemiNet(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level):
        super(PWCSemiNet, self).__init__()
        self.device = device
        self.lv_chs = lv_chs
        self.num_levels = len(lv_chs)
        self.search_range = search_range
        self.batch_norm = batch_norm
        self.corr_activation = corr_activation
        self.residual = residual
        self.output_level = output_level

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs, batch_norm).to(device)
        
        self.warping_layer = WarpingLayer(device)

        # if args.corr == 'CostVolumeLayer':
        #     self.corr = CostVolumeLayer(device, search_range)
        # else:
        #     self.corr = Correlation(pad_size = search_range, kernel_size = 1, max_displacement = search_range, stride1 = 1, stride2 = 1, corr_multiply = 1).to(device)

        self.corr = CostVolumeLayer(device, search_range)
        
        self.flow_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = OpticalFlowEstimator(ch + (search_range*2+1)**2 + (search_range*2+1)**2 + 2, batch_norm).to(device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)
        
        self.context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = ContextNetwork(ch + 2, batch_norm).to(device)
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
        flows = []

        # tensors for summary
        summaries = {
            'x2_warps': [],

        }

        # for x1 in x1_pyramid:
            # print(x1.shape)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 2
                flow_f = torch.zeros(shape).to(self.device)
                flow_b = torch.zeros(shape).to(self.device)
            else:
                output_spatial_size = [x2.size(2), x2.size(3)]
                # flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                flow_f = F.interpolate(flow_f, output_spatial_size, mode='bilinear', align_corners=True) * 2 
                flow_b = F.interpolate(flow_b, output_spatial_size, mode='bilinear', align_corners=True) * 2 
            # print(x2.shape)
            # print(flow.shape)
            x2_semiwarp = self.warping_layer(x2, flow_f / 2)
            x1_semiwarp = self.warping_layer(x1, flow_b / 2)
            x2_warp = self.warping_layer(x2, flow_f)
            x1_warp = self.warping_layer(x1, flow_b)
            
            # correlation
            corr = self.corr(x1_semiwarp, x2_semiwarp)
            corr_f = self.corr(x1, x2_warp)
            corr_b = self.corr(x1_warp, x2)

            if self.corr_activation: 
                F.leaky_relu_(corr)
                F.leaky_relu_(corr_f)
                F.leaky_relu_(corr_b)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                flow_f_coarse = self.flow_estimators[l](torch.cat([x1, corr_f, corr, flow_f], dim = 1)) + flow_f
                flow_b_coarse = self.flow_estimators[l](torch.cat([x2, corr_b, corr, flow_b], dim = 1)) + flow_b
            else:
                flow_f_coarse = self.flow_estimators[l](torch.cat([x1, corr_f, corr, flow_f], dim = 1))
                flow_b_coarse = self.flow_estimators[l](torch.cat([x2, corr_b, corr, flow_b], dim = 1))

            
            flow_f_fine = self.context_networks[l](torch.cat([x1, flow_f], dim = 1))
            flow_b_fine = self.context_networks[l](torch.cat([x2, flow_b], dim = 1))
            flow_f = flow_f_coarse + flow_f_fine
            flow_b = flow_b_coarse + flow_b_fine


            if l == self.output_level:
                flow_f = F.interpolate(flow_f, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                # flow_b = F.interpolate(flow_b, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                summaries['x2_warps'].append(x2_warp.data)
                break
            else:

                summaries['x2_warps'].append(x2_warp.data)

        return flow_f

import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
# from correlation_package.modules.correlation import Correlation


class PWCBothNet(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level):
        super(PWCBothNet, self).__init__()
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
            layer = OpticalFlowEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
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

    def forward(self, x1_raw, x2_raw, x1_next_raw, x2_next_raw):

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]
        x1_next_pyramid = self.feature_pyramid_extractor(x1_next_raw) + [x1_next_raw]
        x2_next_pyramid = self.feature_pyramid_extractor(x2_next_raw) + [x2_next_raw]


        # outputs
        flows = []

        # tensors for summary
        summaries = {
            'x2_warps': [],

        }


        for l, (rgb_l, rgb_r, rgb_next_l, rgb_next_r) in enumerate(zip(x1_pyramid, x2_pyramid, x1_next_pyramid, x2_next_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(rgb_l.size()); shape[1] = 2
                flow = torch.zeros(shape).to(self.device)
                disp = torch.zeros(shape).to(self.device)
            else:
                output_spatial_size = [rgb_l.size(2), rgb_l.size(3)]
                flow = F.interpolate(flow, output_spatial_size, mode='bilinear', align_corners=True) * 2 
                disp = F.interpolate(disp, output_spatial_size, mode='bilinear', align_corners=True) * 2 

            rgb_l_and_next = torch.cat([rgb_l, rgb_next_l], dim=1)
            rgb_r_and_next = torch.cat([rgb_r, rgb_next_r], dim=1)

            rgb_next_l_warp = self.warping_layer(rgb_next_l, flow)
            rgb_r_and_next_warp = self.warping_layer(rgb_r_and_next, -1 * disp)


            # correlation
            corr_optical = self.corr(rgb_l, rgb_next_l_warp)
            if self.corr_activation: F.leaky_relu_(corr_optical)

            corr_disparity = self.corr(rgb_l_and_next, rgb_r_and_next_warp)
            if self.corr_activation: F.leaky_relu_(corr_disparity)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow


            if self.residual:
                flow_coarse = self.flow_estimators[l](torch.cat([rgb_l, corr_optical, flow], dim = 1)) + flow
                disp_coarse = self.flow_estimators[l](torch.cat([rgb_l + rgb_next_l, corr_disparity, disp], dim = 1)) + disp
            else:
                flow_coarse = self.flow_estimators[l](torch.cat([rgb_l, corr_optical, flow], dim = 1))
                disp_coarse = self.flow_estimators[l](torch.cat([rgb_l + rgb_next_l, corr_disparity, disp], dim = 1))

            
            flow_fine = self.context_networks[l](torch.cat([rgb_l, flow], dim = 1))
            disp_fine = self.context_networks[l](torch.cat([rgb_l + rgb_next_l, disp], dim = 1))
            flow = flow_coarse + flow_fine
            disp = disp_coarse + disp_fine


            if l == self.output_level:
                flow = F.interpolate(flow, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

                disp_t0 = disp[:, 0:1, :, :]
                disp_t1 = disp[:, 1:, :, :]
                disp_t0 = F.interpolate(disp_t0, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                disp_t1 = F.interpolate(disp_t1, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

                break



        return {'flow':flow, 'disp':disp_t0, 'disp_next':disp_t1}
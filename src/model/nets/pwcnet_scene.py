import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (WarpingLayer, FeaturePyramidExtractor, CostVolumeLayer, OpticalFlowEstimator, ContextNetwork)
from lib.pwc_modules.modules import DisparityWarpingLayer, DisparityEstimator, DisparityContextNetwork
from lib.pwc_modules.modules import SceneEstimator, SceneContextNetwork
# from correlation_package.modules.correlation import Correlation


class PWCSCNet(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level):
        super(PWCSCNet, self).__init__()
        self.device = device
        self.lv_chs = lv_chs
        self.num_levels = len(lv_chs)
        self.search_range = search_range
        self.batch_norm = batch_norm
        self.corr_activation = corr_activation
        self.residual = residual
        self.output_level = output_level

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs, batch_norm).to(device)
        
        self.optical_warping_layer = WarpingLayer(device)
        self.disparity_warping_layer = DisparityWarpingLayer(device)

        # if args.corr == 'CostVolumeLayer':
        #     self.corr = CostVolumeLayer(device, search_range)
        # else:
        #     self.corr = Correlation(pad_size = search_range, kernel_size = 1, max_displacement = search_range, stride1 = 1, stride2 = 1, corr_multiply = 1).to(device)

        self.corr = CostVolumeLayer(device, search_range)

        

        self.optical_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = OpticalFlowEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.optical_estimators.append(layer)
        
        self.optical_context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = ContextNetwork(ch + 2, batch_norm).to(device)
            self.add_module(f'FlowContextNetwork(Lv{l})', layer)
            self.optical_context_networks.append(layer)


        self.disparity_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
            layer = DisparityEstimator(ch + (search_range*2+1)**2 + 1, batch_norm).to(device)
            self.add_module(f'DisparityEstimator(Lv{l})', layer)
            self.disparity_estimators.append(layer)
        
        self.disparity_context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityContextNetwork(ch + 2, batch_norm).to(device)
            layer = DisparityContextNetwork(ch + 1, batch_norm).to(device)
            self.add_module(f'DisparityContextNetwork(Lv{l})', layer)
            self.disparity_context_networks.append(layer)

        self.scene_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityEstimator(ch + (search_range*2+1)**2 + 2, batch_norm).to(device)
            layer = SceneEstimator(2 + 1 + 1 + 3, batch_norm).to(device)
            self.add_module(f'SceneEstimator(Lv{l})', layer)
            self.scene_estimators.append(layer)

        self.scene_context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            # layer = DisparityContextNetwork(ch + 2, batch_norm).to(device)
            layer = SceneContextNetwork(ch + 3, batch_norm).to(device)
            self.add_module(f'SceneContextNetwork(Lv{l})', layer)
            self.scene_context_networks.append(layer)


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


        for l, (rgb_l, rgb_r, rgb_next_l, rgb_next_r) in enumerate(zip(x1_pyramid, x2_pyramid, x1_next_pyramid, x2_next_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(rgb_l.size()); shape[1] = 2
                flow = torch.zeros(shape).to(self.device)
                shape = list(rgb_l.size()); shape[1] = 1
                disp = torch.zeros(shape).to(self.device)
                disp_next = torch.zeros(shape).to(self.device)
                # shape = list(rgb_l.size()); shape[1] = 3
                # scene_flow = torch.zeros(shape).to(self.device)
            else:
                output_spatial_size = [rgb_l.size(2), rgb_l.size(3)]
                # flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                flow = F.interpolate(flow, output_spatial_size, mode='bilinear', align_corners=True) * 2 
                disp = F.interpolate(disp, output_spatial_size, mode='bilinear', align_corners=True) * 2 
                disp_next = F.interpolate(disp_next, output_spatial_size, mode='bilinear', align_corners=True) * 2 
                # scene_flow = F.interpolate(scene_flow, output_spatial_size, mode='bilinear', align_corners=True) * 2 


            rgb_next_l_warp = self.optical_warping_layer(rgb_next_l, flow)
            optical_corr = self.corr(rgb_l, rgb_next_l_warp)

            rgb_r_warp = self.disparity_warping_layer(rgb_r, disp)
            disp_corr = self.corr(rgb_l, rgb_r_warp)

            rgb_next_r_warp = self.disparity_warping_layer(rgb_next_r, disp_next)
            disp_next_corr = self.corr(rgb_next_l, rgb_next_r_warp)


            if self.corr_activation: F.leaky_relu_(optical_corr)
            if self.corr_activation: F.leaky_relu_(disp_corr)
            if self.corr_activation: F.leaky_relu_(disp_next_corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                flow_coarse = self.optical_estimators[l](torch.cat([rgb_l, optical_corr, flow], dim = 1)) + flow
                disp_coarse = self.disparity_estimators[l](torch.cat([rgb_l, disp_corr, disp], dim = 1)) + disp
                disp_next_coarse = self.disparity_estimators[l](torch.cat([rgb_next_l, disp_next_corr, disp_next], dim = 1)) + disp_next
            else:
                flow_coarse = self.optical_estimators[l](torch.cat([rgb_l, optical_corr, flow], dim = 1))
                disp_coarse = self.disparity_estimators[l](torch.cat([rgb_l, disp_corr, disp], dim = 1))
                disp_next_coarse = self.disparity_estimators[l](torch.cat([rgb_next_l, disp_next_corr, disp_next], dim = 1))

            
            flow_fine = self.optical_context_networks[l](torch.cat([rgb_l, flow], dim = 1))
            disp_fine = self.disparity_context_networks[l](torch.cat([rgb_l, disp], dim = 1))
            disp_next_fine = self.disparity_context_networks[l](torch.cat([rgb_next_l, disp_next], dim = 1))


            flow = flow_coarse + flow_fine
            disp = disp_coarse + disp_fine
            disp_next = disp_next_coarse + disp_next_fine

            # disp_warp = self.optical_warping_layer(disp_next, -1. * flow)


            if l == self.output_level:
                flow = F.interpolate(flow, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                disp = F.interpolate(disp, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                disp_next = F.interpolate(disp_next, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)

                break

        return {'flow':flow, 'disp':disp, 'disp_next':disp_next}

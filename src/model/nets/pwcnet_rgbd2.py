import torch
from time import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.pwc_modules.modules import (WarpingLayer, WarpingLayer3D, FeaturePyramidExtractor, CostVolumeLayer, CostVolumeLayer3D, FlowEstimator3D, ContextNetwork3D, Generate3DFeature)
# from correlation_package.modules.correlation import Correlation


class PWCRGBD2Net(nn.Module):


    def __init__(self, device, lv_chs, search_range, batch_norm, corr_activation, residual, output_level, depth_range):
        super(PWCRGBD2Net, self).__init__()
        self.device = device
        self.lv_chs = lv_chs
        self.num_levels = len(lv_chs)
        self.search_range = search_range
        self.batch_norm = batch_norm
        self.corr_activation = corr_activation
        self.residual = residual
        self.output_level = output_level
        self.depth_range = depth_range

        self.generate_3d_feature = Generate3DFeature(device, depth_range)

        self.feature_pyramid_extractor = FeaturePyramidExtractor(lv_chs, batch_norm).to(device)
        
        self.warping_layer = WarpingLayer3D(device)

        # if args.corr == 'CostVolumeLayer':
        #     self.corr = CostVolumeLayer(device, search_range)
        # else:
        #     self.corr = Correlation(pad_size = search_range, kernel_size = 1, max_displacement = search_range, stride1 = 1, stride2 = 1, corr_multiply = 1).to(device)

        self.corr = CostVolumeLayer3D(device, search_range)

        self.flow_estimators = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = FlowEstimator3D(ch + (search_range*2+1)**3 + 3, batch_norm).to(device)
            self.add_module(f'FlowEstimator(Lv{l})', layer)
            self.flow_estimators.append(layer)
        
        self.context_networks = []
        for l, ch in enumerate(lv_chs[::-1]):
            layer = ContextNetwork3D(ch + 3, batch_norm).to(device)
            self.add_module(f'ContextNetwork(Lv{l})', layer)
            self.context_networks.append(layer)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

            if isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1_raw, x2_raw, x1_disp_raw, x2_disp_raw):

        # on the bottom level are original images
        x1_pyramid = self.feature_pyramid_extractor(x1_raw) + [x1_raw]
        x2_pyramid = self.feature_pyramid_extractor(x2_raw) + [x2_raw]

        # for x1 in x1_pyramid:
            # print(x1.shape)

        for l, (x1, x2) in enumerate(zip(x1_pyramid, x2_pyramid)):
            # upsample flow and scale the displacement
            if l == 0:
                shape = list(x1.size()); shape[1] = 3
                flow = torch.zeros(shape).to(self.device)
                output_spatial_size = [x2.size(2), x2.size(3)]
                x1_disp = F.interpolate(x1_disp_raw, output_spatial_size, mode='bilinear', align_corners=True)
                x2_disp = F.interpolate(x2_disp_raw, output_spatial_size, mode='bilinear', align_corners=True)
            else:
                output_spatial_size = [x2.size(2), x2.size(3)]
                # flow = F.interpolate(flow, scale_factor = 2, mode = 'bilinear', align_corners=True) * 2
                flow = F.interpolate(flow, output_spatial_size, mode='bilinear', align_corners=True) * 2
                x1_disp = F.interpolate(x1_disp_raw, output_spatial_size, mode='bilinear', align_corners=True)
                x2_disp = F.interpolate(x2_disp_raw, output_spatial_size, mode='bilinear', align_corners=True)

            maxi_1 = torch.max(x1_disp)
            maxi_2 = torch.max(x2_disp)
            mini_1 = torch.min(x1_disp)
            mini_2 = torch.min(x2_disp)
            maxi = maxi_1 if maxi_1 > maxi_2 else maxi_2
            mini = mini_1 if mini_1 < mini_2 else mini_2
            x1_disp = (x1_disp - mini) / (maxi - mini)
            x2_disp = (x2_disp - mini) / (maxi - mini)
            x1_disp = x1_disp - 0.5
            x2_disp = x2_disp - 0.5

            x1 = self.generate_3d_feature(x1, x1_disp)
            x2 = self.generate_3d_feature(x2, x2_disp)

            b, c, d, h, w = x1.size()
            flow_cat = torch.zeros(b, 3, d, h, w).to(self.device)
            for i in range(d):
                flow_cat[:, :, i, :, :] = flow[:, :, :, :]
            flow = flow_cat

            x2_warp = self.warping_layer(x2, flow)
    
            # correlation
            corr = self.corr(x1, x2_warp)
            if self.corr_activation: F.leaky_relu_(corr)

            # concat and estimate flow
            # ATTENTION: `+ flow` makes flow estimator learn to estimate residual flow
            if self.residual:
                flow_coarse = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1)) + flow
            else:
                flow_coarse = self.flow_estimators[l](torch.cat([x1, corr, flow], dim = 1))
 
            flow_fine = self.context_networks[l](torch.cat([x1, flow], dim = 1))
            flow = flow_coarse + flow_fine

            # flow = torch.max(flow, 2)
            flow = torch.mean(flow, 2).squeeze_()

            if l == self.output_level:
                flow = F.interpolate(flow, scale_factor = 2 ** (self.num_levels - self.output_level - 1), mode = 'bilinear', align_corners=True) * 2 ** (self.num_levels - self.output_level - 1)
                break

        return {'flow': flow}

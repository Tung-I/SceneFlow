import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import sys

from lib.pwc_modules.utils import get_grid


def conv(batch_norm, in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


class WarpingLayer(nn.Module):
    
    def __init__(self, device):
        super(WarpingLayer, self).__init__()
        self.device = device
    
    def forward(self, x, flow):
        device = self.device
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        # print(get_grid(x).shape)
        # print(flow_for_grip.shape)

        grid = (get_grid(x).to(self.device) + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


class CostVolumeLayer(nn.Module):

    def __init__(self, device, search_range):
        super(CostVolumeLayer, self).__init__()
        self.device = device
        self.search_range = search_range

    
    def forward(self, x1, x2):
        search_range = self.search_range
        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).to(self.device)

        for i in range(-search_range, search_range + 1):
            for j in range(-search_range, search_range + 1):
                if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:       slice_h, slice_h_r = slice(None),    slice(None)

                if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:       slice_w, slice_w_r = slice(None),    slice(None)

                cv[:, (search_range*2+1) * i + j, slice_h, slice_w] = (x1[:,:,slice_h, slice_w]  * x2[:,:,slice_h_r, slice_w_r]).sum(1)
    
        return cv / shape[1]


class FeaturePyramidExtractor(nn.Module):

    def __init__(self, lv_chs, batch_norm):
        super(FeaturePyramidExtractor, self).__init__()
        self.lv_chs = lv_chs
        self.batch_norm = batch_norm

        self.convs = []
        for l, (ch_in, ch_out) in enumerate(zip(self.lv_chs[:-1], self.lv_chs[1:])):
            layer = nn.Sequential(
                conv(self.batch_norm, ch_in, ch_out, stride = 2),
                conv(self.batch_norm, ch_out, ch_out)
            )
            self.add_module(f'Feature(Lv{l})', layer)
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x); feature_pyramid.append(x)

        return feature_pyramid[::-1]
        

class OpticalFlowEstimator(nn.Module):

    def __init__(self, ch_in, batch_norm):
        super(OpticalFlowEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128),
            conv(batch_norm, 128, 128),
            conv(batch_norm, 128, 96),
            conv(batch_norm, 96, 64),
            conv(batch_norm, 64, 32),
            nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        )

    def forward(self, x):
        return self.convs(x)


class ContextNetwork(nn.Module):

    def __init__(self, ch_in, batch_norm):
        super(ContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128, 3, 1, 1),
            conv(batch_norm, 128, 128, 3, 1, 2),
            conv(batch_norm, 128, 128, 3, 1, 4),
            conv(batch_norm, 128, 96, 3, 1, 8),
            conv(batch_norm, 96, 64, 3, 1, 16),
            conv(batch_norm, 64, 32, 3, 1, 1),
            conv(batch_norm, 32, 2, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.convs(x)


class DisparityWarpingLayer(nn.Module):
    
    def __init__(self, device):
        super(DisparityWarpingLayer, self).__init__()
        self.device = device
    
    def forward(self, x, disp):
        device = self.device
        b, c, h, w = disp.size()
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros((b, 2, h, w)).to(self.device)
        flow_for_grip[:,0,:,:] = -1. * disp[:,0,:,:] / ((disp.size(3) - 1.0) / 2.0)
        # flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = 0.

        # print(get_grid(x).shape)
        # print(flow_for_grip.shape)

        grid = (get_grid(x).to(self.device) + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


class DisparityEstimator(nn.Module):

    def __init__(self, ch_in, batch_norm):
        super(DisparityEstimator, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128),
            conv(batch_norm, 128, 128),
            conv(batch_norm, 128, 96),
            conv(batch_norm, 96, 64),
            conv(batch_norm, 64, 32),
            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        )

    def forward(self, x):
        return self.convs(x)


class DisparityContextNetwork(nn.Module):

    def __init__(self, ch_in, batch_norm):
        super(DisparityContextNetwork, self).__init__()

        self.convs = nn.Sequential(
            conv(batch_norm, ch_in, 128, 3, 1, 1),
            conv(batch_norm, 128, 128, 3, 1, 2),
            conv(batch_norm, 128, 128, 3, 1, 4),
            conv(batch_norm, 128, 96, 3, 1, 8),
            conv(batch_norm, 96, 64, 3, 1, 16),
            conv(batch_norm, 64, 32, 3, 1, 1),
            conv(batch_norm, 32, 1, 3, 1, 1)
        )
    
    def forward(self, x):
        return self.convs(x)

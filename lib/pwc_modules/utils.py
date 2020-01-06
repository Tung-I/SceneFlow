import torch

def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)

    return grid

def get_grid_3d(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(4)).view(1, 1, 1, 1, x.size(4)).expand(x.size(0), 1, x.size(2), x.size(3), x.size(4))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3), 1).expand(x.size(0), 1, x.size(2), x.size(3), x.size(4))
    torchDepth = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1, 1).expand(x.size(0), 1, x.size(2), x.size(3), x.size(4))
    grid = torch.cat([torchHorizontal, torchVertical, torchDepth], 1)

    return grid
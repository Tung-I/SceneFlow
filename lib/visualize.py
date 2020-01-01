import numpy as np
import torch
from skimage.color import hsv2rgb

def flow_visualize_2d(optical_flow):

    # input: (C, H, W)

    pi = 3.14159
    h = optical_flow.shape[1]
    w = optical_flow.shape[2]

    flow_degree = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            x_vec = optical_flow[0, i, j]
            y_vec = optical_flow[1, i, j]
            d = np.arctan(y_vec / (x_vec+1e-20))
            if x_vec > 0:
                flow_degree[i, j] = d if y_vec > 0 else (2 * pi - d)
            else :
                flow_degree[i, j] = (pi - d) if y_vec > 0 else (pi + d)

    flow_degree = (flow_degree - flow_degree.min()) / (flow_degree.max()-flow_degree.min())

    flow_img_hsv = np.stack((flow_degree, np.ones((h, w))*0.8, np.ones((h, w))), 2)
    flow_img_rgb = hsv2rgb(flow_img_hsv)

    return np.transpose(flow_img_rgb, (2, 0, 1))

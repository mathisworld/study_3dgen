"""
Ref.: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor

class NeRF(nn.Module):
    def __init__(self, D=8, sdim=256, cdim=128, xdim=3, ddim=3, odim=4, skip_pos=[4], lamb=False):
        """
        D: Depth of network

        sdim: hidden dimension for sigma estimation

        cdim: hidden dimension for color estimation

        xdim: 3D postion

        ddim: 3D viewing direction

        odim: output dimension

        skip_pos: position for skip connection

        lamb: toggle for non-Lambertian
        """
        super().__init__()
        self.D = D
        self.sdim = sdim
        self.cdim = cdim
        self.xdim = xdim
        self.ddim = ddim
        self.odim = odim
        self.skip_pos = skip_pos
        self.lamb = lamb

        self.sigma_linear = nn.ModuleList([nn.Linear(xdim, sdim)] + [nn.Linear(sdim, sdim) if i not in skip_pos else nn.Linear(sdim + xdim, sdim) for i in range(D-1)])
        self.color_linear = nn.Sequential(nn.Linear(ddim + sdim, cdim), nn.ReLU())

        if lamb:
            self.feature_linear = nn.Linear(sdim, sdim)
            self.sigma_est = nn.Linear(sdim, 1)
            self.rgb_est = nn.Linear(cdim, 3)

        else:
            self.output_est = nn.Linear(sdim, odim)

    def forward(self, data:Tensor)->Tensor:
        x, d = torch.split(data, [self.xdim, self.ddim], dim=-1)

        h = x
        for pos, layer in enumerate(self.sigma_linear):
            h = layer(h)
            h = F.relu(h)
            if pos in self.skip_pos:
                h = torch.cat([x, h], dim=-1)

        if self.lamb:
            sigma = self.sigma_est(h)

            feature = self.feature(h)
            h = torch.cat([feature, d], -1)
            color = self.rgb_est(h)

            outputs = torch.cat([color, sigma], -1)

        else:
            outputs = self.output_est(h)

        return outputs



if __name__ == "__main__":
    test_input = torch.randn(256, 6)
    model = NeRF()
    test_output = model(test_input)

    print(test_output.shape)
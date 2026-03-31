"""
Ref.: https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
"""

import torch
import numpy as np

def get_rays(H, W, K, T):
    """
    H: Height of image

    W: Width of image

    K: Camera intrinsic parameter

    T: Camera extrinsic parameter
    """

    # Get image plane coordinate
    """
    u in R^(HxW): Set of x-axis values for HxW sized image
    v in RT(HxW): Set of y-axis values for HxW sized image
    """
    u, v = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy') 

    # Transform: Image Plane to Camera Coordinate 
    """
    Right-handed coordinate
    (OpenCV)
    +x: right direction, +y: down direction, +z: Direction away from the body
    x_cam = (u - cx)/fx, y_cam = (v - cy)/fy. z_cam = +1

    (OpenGL, NeRF) +x: right direction, +y: up direction, +z: Direction pointing to the body
    x_cam = (u - cx)/fx, y_cam = -(v - cy)/fy. z_cam = -1

    d_cam in R^(HxWx3): d_cam[i, j] = (x_cam, y_cam, z_cam)
    """
    d_cam = torch.stack([(u - K[0][2])/K[0][0], -(v - K[1][2])/K[1][1], -torch.ones_like(u)], -1)

    # Transform: Camera Coordinate to World Coordinate 
    """
    T = [R, t] in R^(4x4), R in R^(3x3), t in R^(1x3)
        [0, 1]
    t: location of camera on world coordinate
    R: orientation of camera on world coordinate
    rays_o in R^(HxWx3)
    r = o + t*d
    """
    rays_d = d_cam @ T[:3, :3].T
    rays_o = T[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def fine_sampling(Nc, w, Nf, test=True):
    # Get pdf and cdf
    w = w + 1e-7
    pdf = w / torch.sum(w, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1) #Insert 0 at the beginning

    # Sampling the probability
    if test: # Deterministic sampling for consistent test
        u = torch.linspace(0., 1., steps=Nf)
        u = u.expand(list(cdf.shape[:-1]) + [Nf])

    else: # Uniformly random sampling
        u = torch.rand(list(cdf.shape[:-1]) + [Nf])

    # Inverse transform sampling
    u = u.contiguous() # For torch.searchsorted()
    inds = torch.searchsorted(cdf, u, right=True) # right: Decide whether to return the index to the left or right of the existing value if it duplicates
    below = torch.max(torch.zeros_like(inds-1), inds-1) # find the left of interval
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds) # find the right of interval
    inds_g = torch.stack([below, above], -1)



if __name__ == "__main__":
    K = torch.randn((3,3))
    T = torch.randn((4,4))
    get_rays(128, 256, K, T)

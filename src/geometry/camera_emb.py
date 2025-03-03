from einops import rearrange

from .projection import sample_image_grid, get_local_rays
from ..misc.sht import rsh_cart_2, rsh_cart_4, rsh_cart_6, rsh_cart_8

import torch
import torch.nn.functional as F

def get_intrinsic_embedding(context, degree=0, downsample=1, merge_hw=False):
    assert degree in [0, 2, 4, 8]

    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_local_rays(xy_ray, rearrange(context["intrinsics"], "b v i j -> b v () () i j"),)

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "b v h w d -> b v (h w) d")
    else:
        directions = rearrange(directions, "b v h w d -> b v d h w")

    return directions


def build_rays_torch(c2ws, ixts, H, W, scale=1.0):
    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale
    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    XYZ = torch.cat((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, torch.ones_like(X[:, :, None])), dim=-1).to(c2ws)
    
    i2ws = torch.inverse(ixts).permute(0,2,1) @ c2ws[:,:3, :3].permute(0,2,1)
    XYZ = torch.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(1,H,1,1)
    rays_o = rays_o.repeat(1,1,W,1)
    rays = torch.cat((rays_o, XYZ), dim=-1)
    return rays



def build_plucker_relative(extrinsics, intrinsics, H, W, scale=1.0):
    """
        extrinsics: [B, V, 4, 4] Camera-to-world transformation matrices
        intrinsics: [B, V, 3, 3] Camera intrinsic matrices
        H: Image height
        W: Image width
        scale: Scale factor for image resolution
        
    Returns:
        rays_plucker: [B, V, H, W, 6] Plücker coordinates relative to first view
    """
    B, V = extrinsics.shape[:2]    
    # 1. 首先得到每个视角下的rays (origins和directions)
    extrinsics = extrinsics.reshape(-1, 4, 4)    # [B*V, 4, 4]
    intrinsics = intrinsics.reshape(-1, 3, 3)    # [B*V, 3, 3]
    rays = build_rays_torch(extrinsics, intrinsics, H, W, scale)  # [B, V, H, W, 6]
    return rays.reshape(B, V, int(H*scale),int(W * scale), 6)
    
    # # 2. 将所有rays转换到第一个视角的坐标系下
    # # 获取第一个视角到世界坐标系的变换矩阵和其逆矩阵
    # ref_c2w = extrinsics[:, 0]  # [B, 4, 4]
    # ref_w2c = torch.inverse(ref_c2w)  # [B, 4, 4]
    
    # # 分离rays的origins和directions
    # rays_o = rays[..., :3]  # [B, V, H, W, 3]
    # rays_d = rays[..., 3:]  # [B, V, H, W, 3]
    
    # # 将origins从世界坐标系转换到参考视角的相机坐标系
    # rays_o = rays_o.reshape(B, V*H*W, 3)
    # rays_o_homo = torch.cat([rays_o, torch.ones_like(rays_o[..., :1])], dim=-1)  # [B, V*H*W, 4]
    # rays_o_ref = (ref_w2c[:, None] @ rays_o_homo[..., None]).squeeze(-1)[..., :3]  # [B, V*H*W, 3]
    # rays_o_ref = rays_o_ref.reshape(B, V, H, W, 3)
    
    # # 将directions从世界坐标系转换到参考视角的相机坐标系
    # rays_d = rays_d.reshape(B, V*H*W, 3)
    # rays_d_homo = torch.cat([rays_d, torch.zeros_like(rays_d[..., :1])], dim=-1)  # [B, V*H*W, 4]
    # rays_d_ref = (ref_w2c[:, None] @ rays_d_homo[..., None]).squeeze(-1)[..., :3]  # [B, V*H*W, 3]
    # rays_d_ref = rays_d_ref.reshape(B, V, H, W, 3)
    
    # # 3. 计算Plücker坐标 (momentum = cross(origin, direction))
    # momentum = torch.cross(rays_o_ref, rays_d_ref, dim=-1)
    
    # # 4. 组合direction和momentum为Plücker坐标
    # rays_plucker = torch.cat([rays_d_ref, momentum], dim=-1)  # [B, V, H, W, 6]
    
    # return rays_plucker
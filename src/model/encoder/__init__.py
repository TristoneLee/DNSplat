from dataclasses import dataclass
from typing import Literal, Optional

from einops import rearrange
import torch.nn.functional as F

from .backbone import BackboneCfg
from .common.gaussian_adapter import GaussianAdapterCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .encoder import Encoder
from .visualization.encoder_visualizer import EncoderVisualizer



inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["noposplat", "noposplat_multi","dnsplat"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    

EncoderCfg = EncoderNoPoSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    from .encoder_noposplat import  EncoderNoPoSplat
    from .encoder_dnsplat import EncoderDnSplat
    from .encoder_noposplat_multi import EncoderNoPoSplatMulti
    ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
    "dnsplat": (EncoderDnSplat, None),
}
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer



def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat
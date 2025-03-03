from einops import rearrange
import torch
from torch import nn

from . import BackboneCrocoCfg

from ....misc.cam_utils import update_pose

from .croco.blocks import  Block, FFN
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params
from .croco.patch_embed import get_patch_embed
from ....geometry.camera_emb import build_plucker_relative, build_rays_torch, get_intrinsic_embedding
from . import croco_params

class DnCroCo(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)
            
        # [1, 4, 4]
        self.register_buffer("canonical_camera_extrinsics",torch.tensor([[
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        ]], dtype=torch.float32, requires_grad=False))
        
        self.img_size = self.croco_args['img_size']
        
        self.pluck_embedder = nn.Linear(6,self.dec_embed_dim)
        
        self.extrinsics_decoder = FFN(self.dec_embed_dim,self.dec_embed_dim,6,3) 

        self.set_freeze('encoder')

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        # if not any(k.startswith('dec_blocks2') for k in ckpt):
        #     for key, value in ckpt.items():
        #         if key.startswith('dec_blocks'):
        #             new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks,  self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # image [B, V ,C, H, W]
        assert len(image.shape) == 5
        B,V,C,H,W = image.shape
        image = image.view(-1, *image.shape[2:])
        true_shape = true_shape.view(-1, *true_shape.shape[2:])
        intrinsics_embed = intrinsics_embed.view(-1, 1, intrinsics_embed.shape[-1]) if intrinsics_embed is not None else None
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # x [B*V, Num_patches, D]
        # pos [B*V, Num_patches, 2]

        if intrinsics_embed is not None:
            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x.view(B,V,*x.shape[1:]), pos.view(B,V,*pos.shape[1:])


    def _decoder(self, f, pos, context):
        # project to decoder dim
        # f [B, V, Num_patches, D]
        f = self.decoder_embed(f)
        pos = rearrange(pos, "b v n d -> b (v n) d")
        final_output = []
        
        intrinsics = context['intrinsics']
        pred_extrinsics = []
        
        extrinsics = self.canonical_camera_extrinsics[None].repeat(f.shape[0],f.shape[1],1,1).to(f.device)
        
        for blk in self.dec_blocks:
            trans_delta,rot_delta = torch.chunk(self.extrinsics_decoder(f.mean(dim = 2, keepdim = False)), 2, dim = -1)
            extrinsics = update_pose(trans_delta,rot_delta,extrinsics)
            pred_extrinsics.append(extrinsics)
            extrinsics[:,] = extrinsics[:,0] - extrinsics[:,0].data + self.canonical_camera_extrinsics.repeat(f.shape[0],1,1).to(f.device)
            plucker_ray= build_plucker_relative(extrinsics, intrinsics,256, 256, scale=1.0/16)
            plucker_ray = rearrange(plucker_ray, "b v h w d -> b v (h w) d")
            ray_emb = torch.cat([self.pluck_embedder(plucker_ray),torch.zeros(f.shape[0],f.shape[1],1,f.shape[3]).to(f.device)],dim=2)
            f =  f + ray_emb
            f = rearrange(f, "b v n d -> b (v n) d")
            f = blk(f, pos)
            f = rearrange(f, "b (v n) d -> b v n d", v = f.shape[1] // pos.shape[1])
            # store the result
            final_output.append((f))

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            for i in range(len(final_output)):
                final_output[i] = final_output[i][:,:,:-1, :]

        
        # normalize last output
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        rot_delta, trans_delta = torch.chunk(self.extrinsics_decoder(f.mean(dim = 2, keepdim = False)), 2, dim = -1)
        extrinsics = update_pose(trans_delta,rot_delta,extrinsics)
        pred_extrinsics.append(extrinsics)
        return final_output, pred_extrinsics

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self,
                context: dict,
                return_views=False,
                ):
        ret_dict = {}
        b, v, _, h, w = context["image"].shape
        context.update({'shape':[h,w]})
        device = context["image"].device
        
        view = {'img': context["image"]}

        # camera embedding in the encoder
        # intrinsic embedding [B, V, D, H, W]
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            view['img'] = torch.cat((view['img'], intrinsic_emb), dim=2)

        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            view['intrinsics_embed'] = intrinsic_embedding

        true_shape = view.get('true_shape', torch.tensor(view['img'].shape[-2:])[None][None].repeat(b,v,1,1))
        feat, pos = self._encode_image(view['img'], true_shape, view.get('intrinsics_embed', None))
        
        dec, pred_extrinsics = self._decoder(feat, pos, context)
                
        ret_dict.update({'dec': dec})
        ret_dict.update({'shape': true_shape})
        ret_dict.update({'pred_extrinsics': pred_extrinsics})

        if return_views:
            ret_dict.update('views',view)
        return ret_dict

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024

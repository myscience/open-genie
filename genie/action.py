from typing import Tuple
from torch import Tensor
import torch.nn as nn

from genie.module.attention import SpaceTimeAttention
from genie.module.norm import AdaptiveGroupNorm
from genie.module.quantization import LookupFreeQuantization
from genie.module.video import CausalConv3d, DepthToSpaceTimeUpsample, DepthToSpaceUpsample, DepthToTimeUpsample, SpaceTimeDownsample
from genie.utils import default, enc2dec_name, exists

def get_module(name : str) -> nn.Module:
    match name:
        case 'space-time_attn':
            return SpaceTimeAttention
        case 'space_upsample':
            return DepthToSpaceUpsample
        case 'time_upsample':
            return DepthToTimeUpsample
        case 'spacetime_upsample':
            return DepthToSpaceTimeUpsample
        case 'spacetime_downsample':
            return SpaceTimeDownsample
        case 'adaptive_group_norm':
            return AdaptiveGroupNorm
        case _:
            raise ValueError(f'Unknown module name: {name}')

class LatentAction(nn.Module):
    '''Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    '''
    
    def __init__(
        self,
        blueprint: Tuple[str  | Tuple[str, dict], ...],
        d_codebook: int,
        inp_channels: int = 3,
        ker_size : int | Tuple[int, int] = 3,
        n_embd: int = 256,
        n_codebook: int = 1,
        lfq_bias : bool = True,
        lfq_frac_sample : float = 1.,
        lfq_commit_weight : float = 0.25,
        lfq_entropy_weight : float = 0.1,
        lfq_diversity_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        self.proj_in = CausalConv3d(
            inp_channels,
            out_channels=n_embd,
            kernel_size=ker_size
        )
        
        self.proj_out = CausalConv3d(
            n_embd,
            out_channels=inp_channels,
            kernel_size=ker_size
        )
        
        # Build the encoder and decoder based on the blueprint
        self.enc_layers = nn.ModuleList([])
        self.dec_layers = nn.ModuleList([])
        self.enc_ext = list()
        self.dec_ext = list()
        
        for enc_l in (blueprint):
            # We build a mirror decoder based on the encoder blueprint
            dec_l = enc2dec_name(enc_l)
            
            if isinstance(enc_l, str): enc_l = (enc_l, {})
            if isinstance(dec_l, str): dec_l = (dec_l, {})
            
            name, kwargs = default(enc_l, (None, {}))
            self.enc_ext.extend(
                [kwargs.pop('has_ext', False)] * kwargs.get('n_rep', 1)
            )
            self.enc_layers.extend(
                [
                    get_module(name)(**kwargs)
                    for _ in range(kwargs.pop('n_rep', 1))
                    if exists(name) and exists(kwargs)
                ]
            )
            
            name, kwargs = default(dec_l, (None, {}))
            self.dec_ext.extend(
                [kwargs.pop('has_ext', False)] * kwargs.get('n_rep', 1)
            )
            self.dec_layers.extend(
                [
                    get_module(name)(**kwargs)
                    for _ in range(kwargs.pop('n_rep', 1))
                    if exists(name) and exists(kwargs)
                ]
            )
        
        # Build the quantization module
        self.quant = LookupFreeQuantization(
            d_codebook       = d_codebook,
            n_codebook       = n_codebook,
            use_bias         = lfq_bias,
            frac_sample      = lfq_frac_sample,
            commit_weight    = lfq_commit_weight,
            entropy_weight   = lfq_entropy_weight,
            diversity_weight = lfq_diversity_weight,
        )
        
    def forward(
        self,
        video: Tensor,
        mask : Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        video = self.proj_in(video)
        
        # Encode the video frames into latent actions
        for enc in self.enc_layers:
            video = enc(video, mask=mask)
        
        # Quantize the latent actions
        act, q_loss = self.quant(video)
        
        # Decode the quantized latent actions
        recon = video
        for dec, has_ext in zip(self.dec_layers, self.dec_ext):
            recon = dec(recon, cond=act if has_ext else None)
        
        recon = self.proj_out(recon)
        
        return recon, q_loss
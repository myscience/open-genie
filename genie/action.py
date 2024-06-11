from typing import Tuple
from torch import Tensor
import torch.nn as nn

from genie.module.misc import ForwardBlock
from genie.module.attention import SpatialAttention
from genie.module.attention import TemporalAttention
from genie.module.quantization import LookupFreeQuantization
from genie.utils import default

class ActionBlock(nn.Module):
    
    def __init__(
        self,
        n_embd : int,
        n_head : int | Tuple[int, int],
        d_head : int | Tuple[int, int],
        hid_dim : int | Tuple[int, int] | None = None,
        bias : bool = False,
        embed : bool = True,
        scale : float | None = None,
        dropout : float = 0.0,
    ) -> None:
        super().__init__()
        
        self.embed = embed
        
        if isinstance(n_head, int):
            n_head = (n_head, n_head)
        if isinstance(d_head, int):
            d_head = (d_head, d_head)
        
        self.space_attn = SpatialAttention(
            n_embd=n_embd,
            n_head=n_head[0],
            d_head=d_head[0],
            bias=bias,
            scale=scale,
            causal=False,
            dropout=dropout,
        )
        
        self.temp_attn = TemporalAttention(
            n_embd=n_embd,
            n_head=n_head[1],
            d_head=d_head[1],
            bias=bias,
            scale=scale,
            causal=False,
            dropout=dropout,
        )
        
        self.ffn = ForwardBlock(
            n_embd,
            out_dim=n_embd,
            hid_dim=hid_dim,
            num_groups=n_head[1],
            bias=bias,
        )
        
    def forward(
        self,
        video : Tensor,
        embed : bool | None = None,
        mask  : Tensor | None = None,
    ) -> Tensor:
        embed = default(embed, self.embed)
        
        # We feed the video first through the spatial attention
        # and then through the temporal attention mechanism.
        # NOTE: Positional embeddings are added within the attention
        video = self.space_attn(video, embed=embed, mask=mask) + video
        video = self.temp_attn (video, embed=embed, mask=mask) + video
        video = self.ffn(video) + video
        
        return video

class LatentAction(nn.Module):
    '''Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    '''
    
    def __init__(
        self,
        num_layers: int,
        d_codebook: int,
        n_embd: int = 256,
        n_head: int | Tuple[int, int]= (4, 4),
        d_head: int | Tuple[int, int]= (32, 32),
        ff_hid_dim: int | Tuple[int, int] | None = None,
        dropout: float = 0.1,
        n_codebook: int = 1,
        lfq_bias : bool = True,
        lfq_frac_sample : float = 1.,
        lfq_commit_weight : float = 0.25,
        lfq_entropy_weight : float = 0.1,
        lfq_diversity_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        # Build the encoder module
        self.encoder = nn.Sequential(*[
            ActionBlock(
                n_embd=n_embd,
                n_head=n_head,
                d_head=d_head,
                hid_dim=ff_hid_dim,
                bias=True,
                embed=l == 0,
                dropout=dropout,
            ) for l in range(num_layers)
        ])
        
        self.decoder = nn.Sequential(*[
            ActionBlock(
                n_embd=n_embd,
                n_head=n_head,
                d_head=d_head,
                hid_dim=ff_hid_dim,
                bias=True,
                embed=l == 0,
                dropout=dropout,
            ) for l in range(num_layers)
        ])
        
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
        # Encode the video frames into latent actions
        latent = self.encoder(video, mask=mask)
        
        # Quantize the latent actions
        quant, q_loss = self.quant(latent)
        
        # Decode the quantized latent actions
        recon = self.decoder(quant, mask=mask)
        
        return recon, q_loss
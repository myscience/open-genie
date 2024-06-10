from typing import Tuple
from torch import Tensor
import torch.nn as nn

from genie.module.misc import ForwardBlock
from genie.module.attention import SpatialAttention
from genie.module.attention import TemporalAttention
from genie.module.quantization import LookupFreeQuantization

class ActionBlock(nn.Module):
    
    def __init__(
        self,
        n_embd : int,
        n_head : int | Tuple[int, int],
        d_head : int | Tuple[int, int],
        hid_dim : int | Tuple[int, int] | None = None,
        bias : bool = False,
        scale : float | None = None,
        dropout : float = 0.0,
    ) -> None:
        super().__init__()
        
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
        mask  : Tensor | None = None,
    ) -> Tensor:
        
        # We feed the video first through the spatial attention
        # and then through the temporal attention mechanism.
        video = self.space_attn(video) + video
        video = self.temp_attn(video)
        video = self.ffn(video)

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
    ) -> None:
        super().__init__()
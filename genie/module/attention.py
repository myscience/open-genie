import torch.nn as nn

from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from einops import rearrange
from einops import pack, unpack
from einops.layers.torch import Rearrange

from genie.utils import default

# Inspired by the two cool repos implementations at:
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
# https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/magvit2_pytorch.py#L255
class SelfAttention(nn.Module):
    '''
    Standard self-attention module as originally introduced
    in the paper "Attention is All You Need". This module
    uses the flash-attention implementation offered by
    PyTorch >= 2.0.
    '''
    
    def __init__(
        self,
        n_embd : int,
        n_head : int,
        d_head : int,
        bias : bool = False,
        scale : float | None = None,
        causal : bool = False,
        dropout : float = 0.0,
    ) -> None:
        super().__init__()
        
        self.norm = nn.LayerNorm(n_embd)
        
        self.to_qkv = nn.Sequential(
            nn.Linear(n_embd, 3 * (n_head * d_head), bias=bias),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=n_head),
        )
        
        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(n_head * d_head, n_embd, bias=bias),
        )
        
        self.scale = default(scale, n_embd ** -0.5)
        self.causal = causal
        self.dropout = dropout
        
    def forward(
        self,
        seq : Tensor,
        mask : Tensor | None = None,
    ) -> Tensor:
        '''
        Apply self-attention mechanism to the input sequence.

        Args:
            seq (Tensor): Input sequence tensor of shape (batch_size, sequence_length, embedding_size).
            mask (Tensor, optional): Mask tensor of shape (batch_size, sequence_length) indicating which
                elements in the sequence should be masked. Defaults to None.

        Returns:
            Tensor: Output tensor after applying self-attention mechanism of shape
                (batch_size, sequence_length, embedding_size).
        '''
        seq = self.norm(seq)
        
        # Project the input sequence into query, key, and value
        q, k, v = self.to_qkv(seq)
        
        # Compute the self-attention using fast flash-attention
        attn = scaled_dot_product_attention(q, k, v,
            attn_mask=mask,
            is_causal=self.causal,
            dropout_p=self.dropout,
            scale=self.scale,
        )
        
        # Project the output back to the original embedding dimension
        out = self.to_out(attn)
        
        return out

class SpatialAttention(SelfAttention):
    '''
    Attention module that applies self-attention across the
    spatial dimensions of the input tensor, expected to be
    a 5D tensor of shape (batch, feat, time, height, width).
    '''
    
    def forward(
        self,
        video : Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        b, c, t, h, w = video.shape
        
        inp = rearrange(video, 'b c t h w -> (b t) (h w) c')
        out = super().forward(inp, mask)
        
        return rearrange(out, '(b t) (h w) c -> b c t h w', b=b, t=t, h=h, w=w)

class TemporalAttention(SelfAttention):
    '''
    Attention module that applies self-attention across the
    temporal dimension of the input tensor, expected to be
    a 5D tensor of shape (batch, feat, time, height, width).
    '''
    
    def forward(
        self,
        video : Tensor,
        mask : Tensor | None = None
    ) -> Tensor:
        b, c, t, h, w = video.shape
        
        inp = rearrange(video, 'b c t h w -> (b h w) t c')
        out = super().forward(inp, mask)
        
        return rearrange(out, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
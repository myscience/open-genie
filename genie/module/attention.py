from typing import Literal
import torch
import torch.nn as nn
from math import pi
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from einops import einsum, rearrange, repeat
from einops import pack, unpack
from einops.layers.torch import Rearrange

from genie.utils import default, exists

# Adapted from lucidrains/rotary-embedding-torch at:
# https://github.com/lucidrains/rotary-embedding-torch/
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim : int,
        kind: Literal['1d', '2d', 'const'] = '1d',
        theta = 10000,
        max_freq = 10,
        num_freq = 1,
        learned_freq = False,
        interpolate_factor = 1.,
        theta_rescale_factor = 1.,
    ) -> None:
        super().__init__()
        
        theta *= theta_rescale_factor ** (dim / (dim - 2))

        match kind:
            case '1d':
                freq = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
            case '2d':
                freq = torch.linspace(1., max_freq / 2, dim // 2) * pi
            case 'const':
                freq = torch.ones(num_freq).float()
                
        self.freq = nn.Parameter(freq, requires_grad=learned_freq)
        
        assert interpolate_factor >= 1.
        self.interpolate_factor = interpolate_factor
        
        self.default_seq_dim = -2
        
    def forward(
        self,
        seq : Tensor,
        seq_dim : int | None = None,
        offset = 0,
    ) -> Tensor:
        seq_dim = default(seq_dim, self.default_seq_dim)
        seq_len = seq.shape[seq_dim]
        
        freq = self.freq
        
        # Get sequence position
        pos = (torch.arange(seq_len, device=freq.device) + offset) / self.interpolate_factor
        
        freq = einsum(pos, freq, '..., f -> ... f')
        freq = repeat(freq, '... n -> ... (n r)', r = 2)

        if seq_dim == -3: freq = rearrange(freq, 'n d -> n 1 d')
        
        # Apply rotary embedding
        return self.apply(freq, seq, seq_dim = seq_dim)
        
    def apply(
        self,
        freq : Tensor,
        seq  : Tensor,
        start_index : int = 0,
        scale : float = 1.,
        seq_dim : int = -2
    ) -> Tensor:
        dtype = seq.dtype

        if seq.ndim == 3:
            seq_len = seq.shape[seq_dim]
            freq = freq[-seq_len:]

        rot_dim = freq.shape[-1]
        end_index = start_index + rot_dim

        assert rot_dim <= seq.shape[-1], f'feature dimension {seq.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

        t_left, seq, t_right = seq[..., :start_index], seq[..., start_index:end_index], seq[..., end_index:]
        
        seq = (seq * freq.cos() * scale) + (self.rotate_half(seq) * freq.sin() * scale)
        out = torch.cat((t_left, seq, t_right), dim = -1)
        
        return out.type(dtype)
    
    def rotate_half(self, inp : Tensor) -> Tensor:
        inp = rearrange(inp, '... (d r) -> ... d r', r = 2)
        x1, x2 = inp.unbind(dim = -1)
        inp = torch.stack((-x2, x1), dim = -1)
        return rearrange(inp, '... d r -> ... (d r)')
    
    def get_seq_pos(self, seq_len, device, dtype, offset = 0):
        return (torch.arange(seq_len, device = device, dtype = dtype) + offset) / self.interpolate_factor


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
    either an image (4D tensor) or a video (5D tensor).
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
        super().__init__(n_embd, n_head, d_head, bias, scale, causal, dropout)
        
        self.embed = RotaryEmbedding(n_embd, kind='2d')
    
    def forward(
        self,
        video : Tensor,
        mask: Tensor | None = None
    ) -> Tensor:
        b, *_, h, w = video.shape
        
        inp = rearrange(video, 'b c ... h w -> b ... h w c')
        inp, t_ps = pack([inp], '* h w c')
        
        # Add spatial positional encoding
        inp = self.embed(inp)
        
        inp, s_ps = pack([inp], 'b * c')
        
        out = super().forward(inp, mask)
        
        out = unpack(out, s_ps, 'b * c')[0]
        out = unpack(out, t_ps, '* h w c')[0]
        
        return rearrange(out, 'b ... h w c -> b c ... h w', b=b, h=h, w=w)

class TemporalAttention(SelfAttention):
    '''
    Attention module that applies self-attention across the
    temporal dimension of the input tensor, expected to be
    a 5D tensor of shape (batch, feat, time, height, width).
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
        super().__init__(n_embd, n_head, d_head, bias, scale, causal, dropout)
        
        self.embed = RotaryEmbedding(n_embd, kind='1d')
    
    def forward(
        self,
        video : Tensor,
        mask : Tensor | None = None
    ) -> Tensor:
        b, *_, h, w = video.shape
        
        inp = rearrange(video, 'b c t h w -> (b h w) t c')
        
        # Add temporal positional encoding
        inp = self.embed(inp)
        
        out = super().forward(inp, mask)
        
        return rearrange(out, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
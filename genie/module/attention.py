from typing import Literal, Tuple
import torch
import torch.nn as nn
from math import pi
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from einops import einsum, rearrange, repeat
from einops import pack, unpack
from einops.layers.torch import Rearrange

from genie.module.misc import ForwardBlock
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

class Adapter(nn.Module):
    def __init__(
        self,
        qry_dim : int,
        n_head : int,
        d_head : int,        
        key_dim : int | None = None,
        val_dim : int | None = None,
        block : nn.Module | Tuple[nn.Module, ...] = nn.Linear,
        qry_kwargs : dict = {},
        key_kwargs : dict = {},
        val_kwargs : dict = {},
        bias : bool = False,
    ) -> None:
        super().__init__()

        key_dim = default(key_dim, qry_dim)
        val_dim = default(val_dim, key_dim)

        if issubclass(block, nn.Module):
            block = (block, block, block)
        
        self.to_q = block[0](qry_dim, n_head * d_head, bias=bias, **qry_kwargs)
        self.to_k = block[1](key_dim, n_head * d_head, bias=bias, **key_kwargs)
        self.to_v = block[2](val_dim, n_head * d_head, bias=bias, **val_kwargs)
        
        self.n_head = n_head
        
    def forward(
        self,
        qry : Tensor,
        key : Tensor | None = None,
        val : Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        key = default(key, qry)
        val = default(val, key)
        
        q = self.to_q(qry)
        k = self.to_k(key)
        v = self.to_v(val)
        
        qkv, ps = pack([q, k, v], '* n d')
        qkv = rearrange(qkv, 'qkv n (h d) -> qkv h n d', h=self.n_head)
                
        return unpack(qkv, ps, '* n h d')

# Inspired by the two cool repos implementations at:
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
# https://github.com/lucidrains/magvit2-pytorch/blob/main/magvit2_pytorch/magvit2_pytorch.py#L255
class Attention(nn.Module):
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
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.norm = nn.LayerNorm(n_embd)
        self.embed = nn.Identity()
        
        self.to_qkv = Adapter(
            qry_dim=n_embd,
            n_head=n_head,
            d_head=d_head,
            bias=bias,
            **kwargs,
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
        qry : Tensor,
        key : Tensor | None = None,
        val : Tensor | None = None,
        mask : Tensor | None = None,
    ) -> Tensor:
        '''
        Apply self-attention mechanism to the input sequence.

        Args:
            qry (Tensor): Input sequence tensor of shape (batch_size, sequence_length, embedding_size).
            mask (Tensor, optional): Mask tensor of shape (batch_size, sequence_length) indicating which
                elements in the sequence should be masked. Defaults to None.

        Returns:
            Tensor: Output tensor after applying self-attention mechanism of shape
                (batch_size, sequence_length, embedding_size).
        '''
        
        qry = self.norm(qry)
        qry = self.embed(qry)
        
        key = default(key, qry)
        val = default(val, key)
        
        # Project the input sequence into query, key, and value
        q, k, v = self.to_qkv(qry, key, val)
        
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

class SpatialAttention(Attention):
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
        embed : bool = True,
        scale : float | None = None,
        causal : bool = False,
        dropout : float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            n_embd,
            n_head,
            d_head,
            bias,
            scale,
            causal,
            dropout,
            **kwargs,
        )
        
        # Use 2d-rotary embedding for spatial attention
        self.embed = RotaryEmbedding(n_embd, kind='2d') if embed else nn.Identity()
    
    def forward(
        self,
        video : Tensor,
        cond : Tensor | None = None,
        mask: Tensor | None = None
    ) -> Tensor:
        b, c, *t, h, w = video.shape
        
        inp = rearrange(video, 'b c ... h w -> b ... h w c')
        inp, t_ps = pack([inp], '* h w c')        
        inp, s_ps = pack([inp], 'b * c')
        
        # We expect the condition to be space-wise, i.e. of shape (batch, h * w, feat)
        cond = repeat(cond, 'b hw c -> (b t) hw c', t=t if exists(t) else 1) if exists(cond) else None
        
        out = super().forward(
            inp,
            key=cond,
            mask = mask,
        )
        
        out = unpack(out, s_ps, 'b * c')[0]
        out = unpack(out, t_ps, '* h w c')[0]
        
        return rearrange(out, 'b ... h w c -> b c ... h w', b=b, h=h, w=w)

class TemporalAttention(Attention):
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
        embed : bool = True,
        scale : float | None = None,
        causal : bool = False,
        dropout : float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(n_embd,
            n_head,
            d_head,
            bias,
            scale,
            causal,
            dropout,
            **kwargs,
        )
        
        # Use 1d-rotary embedding for temporal attention
        self.embed = RotaryEmbedding(n_embd, kind='1d') if embed else nn.Identity()
    
    def forward(
        self,
        video : Tensor,
        cond : Tensor | None = None,
        mask : Tensor | None = None
    ) -> Tensor:
        b, *_, h, w = video.shape
        
        inp = rearrange(video, 'b c t h w -> (b h w) t c')
        
        # We expect the condition to be time-wise, i.e. of shape (batch, time, feat)
        cond = repeat(cond, 'b t c -> (b h w) t c', h=h, w=w) if exists(cond) else None
        
        out = super().forward(
            inp,
            key=cond,
            mask=mask,
        )
        
        return rearrange(out, '(b h w) t c -> b c t h w', b=b, h=h, w=w)
    
class SpaceTimeAttention(nn.Module):
    
    def __init__(
        self,
        n_embd : int,
        n_head : int | Tuple[int, int],
        d_head : int | Tuple[int, int],
        hid_dim : int | Tuple[int, int] | None = None,
        bias : bool = False,
        embed : bool | Tuple[bool, bool] = True,
        scale : float | None = None,
        dropout : float = 0.0,
        kernel_size : int = 3,
        time_attn_kw : dict = {},
        space_attn_kw : dict = {},
    ) -> None:
        super().__init__()
        
        if isinstance(n_head, int):
            n_head = (n_head, n_head)
        if isinstance(d_head, int):
            d_head = (d_head, d_head)
        if isinstance(embed, bool):
            embed = (embed, embed)
        
        self.space_attn = SpatialAttention(
            n_embd=n_embd,
            n_head=n_head[0],
            d_head=d_head[0],
            bias=bias,
            scale=scale,
            embed=embed[0],
            causal=False,
            dropout=dropout,
            **space_attn_kw,
        )
        
        self.temp_attn = TemporalAttention(
            n_embd=n_embd,
            n_head=n_head[1],
            d_head=d_head[1],
            bias=bias,
            scale=scale,
            embed=embed[1],
            # * Causal attention for temporal attention
            causal=True, 
            dropout=dropout,
            **time_attn_kw,
        )
        
        self.ffn = ForwardBlock(
            n_embd,
            out_dim=n_embd,
            hid_dim=hid_dim,
            num_groups=n_head[1],
            bias=bias,
            block=nn.Conv3d,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        
    def forward(
        self,
        video : Tensor,
        cond  : Tuple[Tensor, Tensor] | Tensor | None = None,
        mask  : Tensor | None = None,
    ) -> Tensor:
        if not isinstance(cond, tuple):
            cond = (cond, cond)
        
        space_cond, time_cond = cond
        
        # We feed the video first through the spatial attention
        # and then through the temporal attention mechanism.
        # NOTE: Positional embeddings are added within the attention
        video = self.space_attn(video, cond=space_cond, mask=mask) + video
        video = self.temp_attn (video, cond=time_cond , mask=mask) + video
        video = self.ffn(video) + video
        
        return video
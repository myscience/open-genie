import torch.nn as nn
from torch import Tensor
from typing import Tuple
from itertools import pairwise
from einops.layers.torch import Rearrange

from math import prod

from genie.module.misc import ForwardBlock
from genie.module.video import CausalConv3d
from genie.module.image import ResidualBlock as ImageResidualBlock
from genie.module.video import ResidualBlock as VideoResidualBlock

from genie.module.attention import SpatialAttention
from genie.utils import default

class FrameDiscriminator(nn.Module):
    
    def __init__(
        self,
        inp_size : int | Tuple[int, int],
        model_dim : int = 64,
        dim_mults : Tuple[int, ...] = (1, 2, 4),
        down_step : Tuple[int | None, ...] = (None, 2, 2),
        inp_channels : int = 3,
        kernel_size : int | Tuple[int, int] = 3,
        num_groups : int = 1,
        num_heads : int = 4,
        dim_head : int = 32,
        use_attn : bool = False,
        use_blur : bool = True,
        act_fn : str = 'leaky',
    ) -> None:
        super().__init__()
        
        if isinstance(inp_size, int):
            inp_size = (inp_size, inp_size)
            
        # Assemble model core based on dimension schematics
        dims = [model_dim * mult for mult in dim_mults]
        
        assert len(dims) == len(down_step), "Dimension and downsample steps must match."
        
        self.proj_in = nn.Conv2d(
            inp_channels,
            model_dim,
            kernel_size=3,
            padding=1,
        )
        
        self.core = nn.ModuleList([])
        
        for (inp_dim, out_dim), down in zip(pairwise(dims), down_step):
            res_block = ImageResidualBlock(
                inp_dim,
                out_dim,
                downsample=down,
                num_groups=num_groups,
                kernel_size=kernel_size,
            )
            
            attn_block = nn.ModuleList([
                SpatialAttention(
                    out_dim,
                    n_head=num_heads,
                    d_head=dim_head,
                ),
                ForwardBlock(
                    in_dim=out_dim,
                    hid_dim=4 * out_dim,
                    block=nn.Conv2d,
                    kernel_size=1,
                )
            ]) if use_attn else nn.ModuleList([
                nn.Identity(),
                nn.Identity(),
            ])
            
            self.core.append(nn.ModuleList(
                [
                    res_block,
                    attn_block
                ]
            ))
            
            inp_size = tuple(map(lambda x: x // (down or 1), inp_size))
            
        # Compute latent dimension as the product of the last dimension and the frame size
        latent_dim = out_dim * prod(inp_size)
        
        self.to_logits = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )
        
    def forward(
        self,
        image : Tensor,
    ) -> Tensor:
        
        out = self.proj_in(image)
        
        for res, (attn, ff) in self.core:
            # Apply residual block
            out = res(out)
            
            # Apply attention block
            out = attn(out) + out
            out =   ff(out) + out
        
        return self.to_logits(out)
    
class VideoDiscriminator(nn.Module):
    
    def __init__(
        self,
        inp_size : Tuple[int, int] | Tuple[int, int, int],
        model_dim : int = 64,
        dim_mults : Tuple[int, ...] = (1, 2, 4),
        down_step : Tuple[int | Tuple[int, int] | None, ...] = (None, 2, 2),
        inp_channels : int = 3,
        kernel_size : int | Tuple[int, int] = 3,
        num_groups : int = 1,
        num_heads : int = 4,
        dim_head : int = 32,
        act_fn : str = 'leaky',
        use_attn : bool = False,
        use_blur : bool = True,
        use_causal : bool = False,
    ) -> None:
        super().__init__()
        
        if len(inp_size) == 2:
            inp_size = (inp_size[0], inp_size[1], inp_size[1])
        
        Conv3d = CausalConv3d if use_causal else nn.Conv3d
        
        # Assemble model core based on dimension schematics
        dims = [model_dim * mult for mult in dim_mults]
        
        assert len(dims) == len(down_step), "Dimension and downsample steps must match."
        
        self.proj_in = Conv3d(
            inp_channels,
            model_dim,
            kernel_size=kernel_size,
            padding=1,
        )
        
        self.core = nn.ModuleList([])
        
        for (inp_dim, out_dim), down in zip(pairwise(dims), down_step):
            res_block = VideoResidualBlock(
                inp_dim,
                out_dim,
                downsample=down,
                num_groups=num_groups,
                kernel_size=kernel_size,
                act_fn=act_fn,
                use_blur=use_blur,
                use_causal=use_causal,
            )
            
            attn_block = nn.ModuleList([
                SpatialAttention(
                    out_dim,
                    n_head=num_heads,
                    d_head=dim_head,
                ),
                ForwardBlock(
                    in_dim=out_dim,
                    hid_dim=4 * out_dim,
                    block=Conv3d,
                    kernel_size=1,
                )
            ]) if use_attn else nn.ModuleList([
                nn.Identity(),
                nn.Identity(),
            ])
            
            self.core.append(nn.ModuleList(
                [
                    res_block,
                    attn_block
                ]
            ))
            
            down = default(down, (1, 1, 1))
            if isinstance(down, int): down = (down, down, down)
            if len(down) == 2: down = (down[0], down[1], down[1])
            inp_size = tuple(map(lambda x, y: x // y, inp_size, down))
        
        # Compute latent dimension as the product of the last dimension and the frame size
        latent_dim = out_dim * prod(inp_size)
        
        self.to_logits = nn.Sequential(
            nn.Conv3d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )
        
    def forward(
        self,
        image : Tensor,
    ) -> Tensor:
        
        out = self.proj_in(image)
        
        for res, (attn, ff) in self.core:
            # Apply residual block
            out = res(out)
            
            # Apply attention block
            out = attn(out) + out
            out =   ff(out) + out
        
        return self.to_logits(out)
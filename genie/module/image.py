from itertools import pairwise
from torch import Tensor
import torch.nn as nn

from typing import Tuple

from einops.layers.torch import Rearrange

from genie.module.attention import SpatialAttention
from genie.module.misc import ForwardBlock
from genie.utils import exists
from genie.utils import default

class SpaceDownsample(nn.Module):
    def __init__(
        self,
        in_dim : int,
        factor : int = 2,
    ) -> None:
        super().__init__()
        
        self.go_up = nn.Sequential(
            Rearrange('b c (h p) (w q) -> b (c p q) h w', p=factor, q=factor),
            nn.Conv2d(in_dim * factor ** 2, in_dim, kernel_size=1),
        )
        
    def forward(
        self,
        inp : Tensor,
    ) -> Tensor:
        return self.go_up(inp)

class ResidualBlock(nn.Module):
    
    def __init__(
        self,
        inp_channel : int,
        out_channel : int | None = None,
        kernel_size : int | Tuple[int, int] = 3,
        padding : int | Tuple[int, int] = 1,
        num_groups : int = 1,
        downsample : int | None = None,
    ) -> None:
        super().__init__()
        
        self.res = nn.Conv2d(
            inp_channel,
            out_channel,
            kernel_size=1,
            stride=default(downsample, 1),
        ) if exists(out_channel) else nn.Identity()
        
        out_channel = default(out_channel, inp_channel)
        
        self.main = nn.Sequential(
            nn.GroupNorm(num_groups, inp_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                inp_channel,
                out_channel,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.GroupNorm(num_groups, out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
                padding=padding,
            ),
            *(
                [SpaceDownsample(out_channel, downsample)]
                if exists(downsample) and downsample
                else []
            )
        )
        
    def forward(
        self,
        inp : Tensor,
    ) -> Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            inp (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor after applying the residual block operations.
        """
        return self.main(inp) + self.res(inp)

class FrameDiscriminator(nn.Module):
    
    def __init__(
        self,
        frame_size : int | Tuple[int, int],
        model_dim : int = 64,
        dim_mults : Tuple[int, ...] = (1, 2, 4),
        down_step : Tuple[int | None, ...] = (None, 2, 2),
        inp_channels : int = 3,
        kernel_size : int | Tuple[int, int] = 3,
        num_groups : int = 1,
        num_heads : int = 4,
        dim_head : int = 32,
    ) -> None:
        super().__init__()
        
        if isinstance(frame_size, int):
            frame_size = (frame_size, frame_size)
            
        # Assemble model core based on dimension schematics
        dims = [inp_channels] + [model_dim * mult for mult in dim_mults]
        
        assert len(dims) - 1 == len(down_step), "Dimension and downsample steps must match."
        
        self.core = nn.ModuleList([])
        
        for (inp_dim, out_dim), down in zip(pairwise(dims), down_step):
            res_block = ResidualBlock(
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
            ])
            
            self.core.append(nn.ModuleList(
                [
                    res_block,
                    attn_block
                ]
            ))
            
            frame_size = tuple(map(lambda x: x // (down or 1), frame_size))
            
        # Compute latent dimension as the product of the last dimension and the frame size
        latent_dim = out_dim * frame_size[0] * frame_size[1]
        
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
        
        out = image
        
        for res, (attn, ff) in self.core:
            # Apply residual block
            out = res(out)
            
            # Apply attention block
            out = attn(out) + out
            out =   ff(out) + out
        
        return self.to_logits(out)
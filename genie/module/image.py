from torch import Tensor
import torch.nn as nn

from typing import Tuple

from einops.layers.torch import Rearrange

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
        num_groups : int = 1,
        downsample : int | None = None,
    ) -> None:
        super().__init__()
        
        self.res = nn.Conv3d(
            inp_channel,
            out_channel,
            kernel_size=1
        ) if exists(out_channel) else nn.Identity()
        
        out_channel = default(out_channel, inp_channel)
        
        self.main = nn.Sequential(
            nn.GroupNorm(num_groups, inp_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                inp_channel,
                out_channel,
                kernel_size=kernel_size,
            ),
            nn.GroupNorm(num_groups, out_channel),
            nn.LeakyReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=kernel_size,
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
        inp_channels : int = 3,
    ) -> None:
        super().__init__()
        
        self.core = nn.Sequential(
        )
        
        self.to_logits = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            Rearrange('b ... -> b (...)'),
            nn.Linear(latent_dim, 1),
            Rearrange('b 1 -> b')
        )
        
    def forward(
        self,
        inp : Tensor,
    ) -> Tensor:
        
        for layer in self.core:
            inp = layer(inp)
        
        return self.to_logits(inp)
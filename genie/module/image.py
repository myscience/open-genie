import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from math import comb

from torch.types import Device

from torch.nn.functional import conv2d

from einops import repeat
from einops.layers.torch import Rearrange

from genie.utils import exists
from genie.utils import default

def get_blur_kernel(
    kernel_size : int | Tuple[int, int],
    device : Device = None,
    dtype : torch.dtype | None = None,
    norm : bool = True
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # Construct the 1d pascal blur kernel
    ker_a_1d = torch.tensor(
        [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    ).unsqueeze(-1)
    ker_b_1d = torch.tensor(
        [comb(kernel_size[1] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    ).unsqueeze(0)
    

    ker_2d = ker_a_1d @ ker_b_1d
    
    return ker_2d / ker_2d.sum() if norm else ker_2d

# Inspired by the (very cool) kornia library, see the original implementation here:
# https://github.com/kornia/kornia/blob/e461f92ff9ee035d2de2513859bee4069356bc25/kornia/filters/blur_pool.py#L21
class BlurPooling2d(nn.Module):
    def __init__(
        self,
        kernel_size : int | Tuple[int, int],
        # Expected kwargs are the same as the one accepted by Conv2d
        stride : int | Tuple[int, int] = 2,
        num_groups : int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        
        # Register the blurring kernel buffer
        self.register_buffer('blur', get_blur_kernel(kernel_size))
        
        self.stride = stride
        self.kwargs = kwargs
        self.num_groups = num_groups
        
        str_h, str_w = stride      if isinstance(stride,      tuple) else (stride, stride)
        ker_h, ker_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = (ker_h - 1) // str_h, (ker_w - 1) // str_w
        
    def forward(
        self,
        inp : Tensor,
    ) -> Tensor:
        b, c, h, w = inp.shape
        
        # Repeat spatial kernel for each channel of input image
        ker = repeat(self.blur, 'i j -> c g i j', c=c, g=c // self.num_groups)
        
        # Compute the blur as 2d convolution
        return conv2d(
            inp, ker,
            stride=self.stride,
            padding=self.padding,
            groups=self.num_groups,
            **self.kwargs
        )

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
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad

from typing import Tuple
from functools import partial
from einops import rearrange

class CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size : int | Tuple[int, int, int],
        pad_mode : str = 'constant',
        **kwargs
    ):
        super(self).__init__()
        
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        
        # Compute the appropriate causal padding
        stride   = kwargs.pop('stride', 1)
        dilation = kwargs.pop('dilation', 1)
        
        time_ker, height_ker, width_ker = kernel_size
        time_pad = (time_ker - 1) * dilation + (1 - stride)
        height_pad = height_ker // 2
        width_pad  = width_ker  // 2
        
        # Causal padding pads time only to the left to ensure causality
        self.causal_pad = partial(
            pad,
            pad=(width_pad, width_pad, height_pad, height_pad, time_pad, 0),
            mode=pad_mode
        )
        
        self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=(stride, 1, 1),
            dilation=(dilation, 1, 1),
            **kwargs
        )

    def forward(self, inp : Tensor) -> Tensor:
        # Insert causal padding
        inp = self.causal_pad(inp)
        
        return self.conv3d(inp)
    
class CausalConvTranspose3d(nn.ConvTranspose3d):
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        kernel_size : int | Tuple[int, int, int],
        time_stride : int,
        **kwargs,
    ) -> None:
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size, kernel_size)
        _, height_ker, width_ker = kernel_size
        
        super(CausalConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=(time_stride, 1, 1),
            padding=(0, height_ker // 2, width_ker // 2),
            **kwargs,
        )
        
    def forward(self, inp : Tensor) -> Tensor:
        *_, t, h, w = inp.shape
        T = t * self.stride[0]
        
        return super().forward(inp)[..., :T, :, :]
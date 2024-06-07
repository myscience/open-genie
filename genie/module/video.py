import torch.nn as nn
from torch import Tensor
from torch.nn.functional import pad
from einops.layers.torch import Rearrange

from typing import Tuple
from functools import partial
from einops import pack
from einops import unpack
from einops import rearrange

from genie.utils import default, exists

class CausalConv3d(nn.Module):
    """
    3D Causal Convolutional Layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or Tuple[int, int, int]): Size of the convolutional kernel.
        stride (int or Tuple[int, int, int], optional): Stride of the convolution. Defaults to (1, 1, 1).
        dilation (int or Tuple[int, int, int], optional): Dilation rate of the convolution. Defaults to (1, 1, 1).
        pad_mode (str, optional): Padding mode. Defaults to 'constant'.
        **kwargs: Additional keyword arguments to be passed to the nn.Conv3d constructor.

    Attributes:
        causal_pad (partial): Partial function for applying causal padding.
        conv3d (nn.Conv3d): 3D convolutional layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int],
        stride: int | Tuple[int, int, int] = (1, 1, 1),
        dilation: int | Tuple[int, int, int] = (1, 1, 1),
        space_pad : int | Tuple[int, int] | None = None,
        pad_mode: str = 'constant',
        **kwargs
    ):
        super().__init__()

        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(space_pad, int | None):
            space_pad = (space_pad, space_pad)

        t_stride, *s_stride = stride
        t_dilation, *s_dilation = dilation

        # Compute the appropriate causal padding

        time_ker, height_ker, width_ker = kernel_size
        time_pad = (time_ker - 1) * t_dilation + (1 - t_stride)
        height_pad = default(space_pad[0], (height_ker - 1) // 2)
        width_pad  = default(space_pad[1], (width_ker  - 1)  // 2)

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
            stride=(t_stride, *s_stride),
            dilation=(t_dilation, *s_dilation),
            **kwargs
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass of the CausalConv3d layer.

        Args:
            inp (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the CausalConv3d layer.

        """
        # Insert causal padding
        inp = self.causal_pad(inp)

        return self.conv3d(inp)
    
    @property
    def inp_dim(self) -> int:
        return self.in_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels
    
class CausalConvTranspose3d(nn.ConvTranspose3d):
    """
    3D Causal Convolutional Transpose layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or Tuple[int, int, int]): Size of the convolutional kernel.
        stride (int or Tuple[int, int, int], optional): Stride of the convolution. Default is (1, 1, 1).
        dilation (int or Tuple[int, int, int], optional): Dilation rate of the convolution. Default is (1, 1, 1).
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Attributes:
        Same as the parent class `nn.ConvTranspose3d`.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int, int],
        stride  : int | Tuple[int, int, int] = (1, 1, 1),
        dilation: int | Tuple[int, int, int] = (1, 1, 1),
        space_pad : int | Tuple[int, int] | None = None,
        **kwargs,
    ) -> None:
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(space_pad, int | None):
            space_pad = (space_pad, space_pad)
        _, height_ker, width_ker = kernel_size
        
        height_pad = default(space_pad[0], height_ker // 2)
        width_pad  = default(space_pad[1], width_ker  // 2)

        super(CausalConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(0, height_pad, width_pad),
            **kwargs,
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, inp: Tensor) -> Tensor:
        """
        Forward pass of the CausalConvTranspose3d layer.

        Args:
            inp (Tensor): Input tensor of shape (batch_size, in_channels, t, h, w).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, t', h', w').

        """
        *_, t, h, w = inp.shape
        T, H, W = self.stride

        return super().forward(inp)[..., :t * T, :h * H, :w * W]
    
    @property
    def inp_dim(self) -> int:
        return self.in_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels
    
class DepthToSpaceUpsample(nn.Module):
    '''Depth to Space Upsampling module.
    '''
    
    def __init__(
        self,
        in_channels : int, 
        out_channels : int | None = None,
        factor : int = 2,
    ) -> None:
        super().__init__()
        
        out_channels = default(out_channels, in_channels)
    
        self.go_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * factor ** 2, kernel_size=1),
            Rearrange('b (c p q) h w -> b c (h p) (w q)', p=factor, q=factor),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        # Input is expected to be a video, rearrange it to have
        # shape suitable for a Conv2d layer to operate on
        inp = rearrange(inp, 'b c t h w -> b t c h w')
        inp, ps = pack([inp], '* c h w')
        
        out = self.go_up(inp)
        
        # Restore video format
        out, *_ = unpack(out, ps, '* c h w')
        out = rearrange(out, 'b t c h w -> b c t h w')
        
        return out
    
    @property
    def inp_dim(self) -> int:
        return self.in_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels
    
class DepthToTimeUpsample(nn.Module):
    '''Depth to Time Upsampling module.
    '''
    
    def __init__(
        self,
        in_channels : int, 
        out_channels : int | None = None,
        factor : int = 2,
    ) -> None:
        super().__init__()
        
        out_channels = default(out_channels, in_channels)
    
        self.go_up = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * factor, kernel_size=1),
            Rearrange('b (c f) t -> b c (t f)', f=factor),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        # Input is expected to be a video, rearrange it to have
        # shape suitable for a Conv2d layer to operate on
        inp = rearrange(inp, 'b c t h w -> b h w c t')
        inp, ps = pack([inp], '* c t')
        
        out = self.go_up(inp)
        
        # Restore video format
        out, *_ = unpack(out, ps, '* c t')
        out = rearrange(out, 'b h w c t -> b c t h w')
        
        return out
    
    @property
    def inp_dim(self) -> int:
        return self.in_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels

class DepthToSpaceTimeUpsample(nn.Module):
    '''Depth to Space-Time Upsample
    '''
    def __init__(
        self,
        in_channels : int,
        out_channels : int | None = None, 
        time_factor  : int = 2,
        space_factor : int = 2,
        kernel_size : int | Tuple[int, int, int] = 1,
    ) -> None:
        super().__init__()
        
        out_channels = default(out_channels, in_channels)
    
        self.go_up = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels * time_factor * space_factor ** 2,
                kernel_size=kernel_size,
            ),
            Rearrange(
                'b (c p q r) t h w -> b c (t p) (h q) (w r)',
                p=time_factor,
                q=space_factor,
                r=space_factor
            ),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        # Input is expected to be a video
        out = self.go_up(inp)
        
        return out
    
    @property
    def inp_dim(self) -> int:
        return self.in_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels

class SpaceTimeDownsample(CausalConv3d):
    '''Space-Time Downsample module.
    '''
    
    def __init__(
        self,
        in_channels : int,
        out_channels : int,
        time_factor : int = 2,
        space_factor : int = 2,
        **kwargs
    ) -> None:
        kernel_size = kwargs.pop('kernel_size', (time_factor, space_factor, space_factor))
        super().__init__(
            in_channels,
            out_channels,
            stride = (time_factor, space_factor, space_factor),
            kernel_size = kernel_size,
            **kwargs,
        )

class SpaceTimeUpsample(CausalConvTranspose3d):
    '''Space-Time Upsample module.
    '''
    
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        time_factor : int = 2,
        space_factor : int = 2,
        **kwargs
    ) -> None:
        super().__init__(
            in_dim,
            out_dim,
            stride=(time_factor, space_factor, space_factor),
            kernel_size = (time_factor, space_factor, space_factor),
            space_pad=0,
            **kwargs,
        )


class ResidualBlock(nn.Module):
    """
    A residual block module that performs residual connections and applies convolutional operations.
    
    Args:
        inp_channel (int): The number of input channels.
        out_channel (int | None, optional): The number of output channels. If None, it defaults to inp_channel. 
        kernel_size (int | Tuple[int, int, int], optional): The size of the convolutional kernel. Defaults to 3.
        num_groups (int, optional): The number of groups to separate the channels into for group normalization. Defaults to 32.
        pad_mode (str, optional): The padding mode for convolution. Defaults to 'constant'.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels : int | None = None,
        kernel_size : int | Tuple[int, int, int] = 3,
        num_groups : int = 1,
        pad_mode : str = 'constant',
    ) -> None:
        super().__init__()
        
        self.res = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=1
        ) if exists(out_channels) else nn.Identity()
        
        out_channels = default(out_channels, in_channels)
        
        self.main = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            CausalConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                pad_mode=pad_mode,
            ),
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                pad_mode=pad_mode,
            )
        )
        
        self.inp_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            inp (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor after applying the residual block operations.
        """
        return self.main(inp) + self.res(inp)
    
    @property
    def inp_dim(self) -> int:
        return self.inp_channels
    
    @property
    def out_dim(self) -> int:
        return self.out_channels
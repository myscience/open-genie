import torch
import torch.nn as nn
from abc import ABC
from torch import Tensor
from torch.nn.functional import pad
from torch.nn.functional import conv3d
from einops.layers.torch import Rearrange

from math import comb
from torch.types import Device

from typing import Tuple
from functools import partial
from einops import pack
from einops import unpack
from einops import repeat
from einops import einsum
from einops import rearrange

from genie.utils import default, exists

def get_blur_kernel(
    kernel_size : int | Tuple[int, int],
    device : Device = None,
    dtype : torch.dtype | None = None,
    norm : bool = True
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    
    # Construct the 1d pascal blur kernel
    ker_t_1d = torch.tensor(
        [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
        device=device,
        dtype=dtype,
    )
    ker_h_1d = rearrange(
        torch.tensor(
            [comb(kernel_size[0] - 1, i) for i in range(kernel_size[0])],
            device=device,
            dtype=dtype,
        ),
        'h -> h 1'
    )
    ker_w_1d = rearrange(
            torch.tensor(
            [comb(kernel_size[1] - 1, i) for i in range(kernel_size[0])],
            device=device,
            dtype=dtype,
        ),
        'w -> 1 w'
    )
    
    ker_3d = einsum(ker_t_1d, ker_h_1d @ ker_w_1d, 't, h w -> t h w')
    
    return ker_3d / ker_3d.sum() if norm else ker_3d

class Upsample(nn.Module, ABC):
    def __init__(
        self,
        time_factor : int = 1,
        space_factor : int = 1,
    ) -> None:
        super().__init__()
        
        self.time_factor = time_factor
        self.space_factor = space_factor
        
        self.go_up = None
    
    @property
    def factor(self) -> int:
        return self.time_factor * (self.space_factor ** 2)
    
    def forward(
        self,
        inp : Tensor,
        **kwargs,
    ) -> Tensor:
        return self.go_up(inp)
    
class Downsample(nn.Module, ABC):
    def __init__(
        self,
        time_factor : int = 1,
        space_factor : int = 1,
    ) -> None:
        super().__init__()
        
        self.time_factor = time_factor
        self.space_factor = space_factor
        
        self.go_down = None
    
    @property
    def factor(self) -> int:
        return self.time_factor * (self.space_factor ** 2)
    
    def forward(
        self,
        inp : Tensor,
        **kwargs,
    ) -> Tensor:
        return self.go_down(inp)

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
        padding : int | Tuple[int, int] | None = None,
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
        if isinstance(padding, int | None):
            padding = (padding, padding)

        t_stride, *s_stride = stride
        t_dilation, *s_dilation = dilation

        # Compute the appropriate causal padding
        if isinstance(padding, int | None):
            padding = (padding, padding)

        time_ker, height_ker, width_ker = kernel_size
        time_pad = (time_ker - 1) * t_dilation + (1 - t_stride)
        height_pad = default(padding[0], (height_ker - 1) // 2)
        width_pad  = default(padding[1], (width_ker  - 1) // 2)

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
    
class DepthToSpaceUpsample(Upsample):
    '''Depth to Space Upsampling module.
    '''
    
    def __init__(
        self,
        in_channels : int, 
        out_channels : int | None = None,
        factor : int = 2,
    ) -> None:
        super().__init__(
            space_factor=factor,
        )
        
        out_channels = default(out_channels, in_channels)
    
        self.go_up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * factor ** 2, kernel_size=1),
            Rearrange('b (c p q) h w -> b c (h p) (w q)', p=factor, q=factor),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor,
        **kwargs,
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
    
class DepthToTimeUpsample(Upsample):
    '''Depth to Time Upsampling module.
    '''
    
    def __init__(
        self,
        in_channels : int, 
        out_channels : int | None = None,
        factor : int = 2,
    ) -> None:
        super().__init__(
            time_factor=factor,
        )
        
        out_channels = default(out_channels, in_channels)
    
        self.go_up = nn.Sequential(
            nn.Conv1d(in_channels, out_channels * factor, kernel_size=1),
            Rearrange('b (c f) t -> b c (t f)', f=factor),
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
    def forward(
        self,
        inp : Tensor,
        **kwargs,
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

class DepthToSpaceTimeUpsample(Upsample):
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
        super().__init__(
            time_factor=time_factor,
            space_factor=space_factor,
        )
        
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
        inp : Tensor,
        **kwargs,
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

class SpaceTimeUpsample(Upsample):
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
            time_factor=time_factor,
            space_factor=space_factor,
        )
        
        self.go_up = nn.ConvTranspose3d(
            in_dim,
            out_dim,
            kernel_size=(time_factor, space_factor, space_factor),
            stride=(time_factor, space_factor, space_factor),
            **kwargs,
        )

class SpaceTimeDownsample(Downsample):
    '''Space-Time Downsample module.
    '''
    
    def __init__(
        self,
        in_channels : int,
        kernel_size : int | Tuple[int, int, int],
        out_channels : int | None = None,
        time_factor : int = 2,
        space_factor : int = 2,
        **kwargs
    ) -> None:
        super().__init__(
            time_factor=1 / time_factor,
            space_factor=1 / space_factor,
        )
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        self.go_down = CausalConv3d(
            in_channels,
            default(out_channels, in_channels),
            kernel_size = kernel_size,
            stride = (time_factor, space_factor, space_factor),
            **kwargs,
        )

# Inspired by the (very cool) kornia library, see the original implementation here:
# https://github.com/kornia/kornia/blob/e461f92ff9ee035d2de2513859bee4069356bc25/kornia/filters/blur_pool.py#L21
class BlurPooling3d(nn.Module):
    def __init__(
        self,
        in_channels : int, # Needed only for compatibility
        kernel_size : int | Tuple[int, int, int],
        out_channels : int | None = None,
        time_factor : int = 2,
        space_factor : int | Tuple[int, int] = 2,
        num_groups : int = 1,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(space_factor, int):
            space_factor = (space_factor, space_factor)
        
        # Register the blurring kernel buffer
        self.register_buffer('blur', get_blur_kernel(kernel_size))
        
        self.stride = (time_factor, *space_factor)
        self.kwargs = kwargs
        self.num_groups = num_groups
        self.out_channels = out_channels
        
        ker_t, ker_h, ker_w = kernel_size
        self.padding = (ker_t - 1) // 2, (ker_h - 1) // 2, (ker_w - 1) // 2
        
    def forward(
        self,
        inp : Tensor,
    ) -> Tensor:
        b, c, t, h, w = inp.shape
        
        o = default(self.out_channels, c)
        
        # Repeat spatial kernel for each channel of input image
        ker = repeat(self.blur, 'i j k -> o g i j k', o=o, g=c // self.num_groups)
        
        # Compute the blur as 2d convolution
        return conv3d(
            inp, ker,
            stride=self.stride,
            padding=self.padding,
            groups=self.num_groups,
            **self.kwargs
        )
        
    def __repr__(self):
        return f'BlurPooling3d({self.out_channels}, kernel_size={tuple(self.blur.shape)}, stride={self.stride}, padding={self.padding})'

class ResidualBlock(nn.Module):
    """
    A residual block module that performs residual connections and applies
    convolutional operations, with flexible options for normalization and
    down-sampling of input.
    
    Args:
        inp_channel (int): The number of input channels.
        out_channel (int | None, optional): The number of output channels. If None, it defaults to inp_channel. 
        kernel_size (int | Tuple[int, int, int], optional): The size of the convolutional kernel. Defaults to 3.
        num_groups (int, optional): The number of groups to separate the channels into for group normalization. Defaults to 32.
        pad_mode (str, optional): The padding mode for convolution. Defaults to 'constant'.
        downsample (int | Tuple[int, int] | None, optional): The factor by which to downsample the input. Defaults to None.
        causal (bool, optional): Whether to use a causal convolution. Defaults to False.
        use_norm (bool, optional): Whether to use normalization. Defaults to True.
        use_blur (bool, optional): Whether to use blur pooling. Defaults to True.
        act_fn (str, optional): The activation function to use. Defaults to 'swish'.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels : int | None = None,
        kernel_size : int | Tuple[int, int, int] = 3,
        num_groups : int = 1,
        pad_mode : str = 'constant',
        downsample : int | Tuple[int, int] | None = None,
        use_causal : bool = False,
        use_norm : bool = True,
        use_blur : bool = True,
        act_fn : str = 'swish',
    ) -> None:
        super().__init__()
        
        if isinstance(downsample, int):
            downsample = (downsample, downsample)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        
        Norm = nn.GroupNorm  if use_norm else nn.Identity
        Down = BlurPooling3d if use_blur else SpaceTimeDownsample
        Conv = partial(CausalConv3d, pad_mode=pad_mode) if use_causal else nn.Conv3d
        
        match act_fn:
            case 'relu': Act = nn.ReLU
            case 'gelu': Act = nn.GELU
            case 'leaky': Act = nn.LeakyReLU
            case 'swish' | 'silu': Act = nn.SiLU
        
        out_channels = default(out_channels, in_channels)
        time_factor, space_factor = downsample if exists(downsample) else (None, None)
        
        self.res = nn.Sequential(
            Down(
                in_channels,
                kernel_size,
                time_factor=time_factor,
                space_factor=space_factor,
                num_groups=num_groups,
            ) if exists(downsample) else nn.Identity(),
            Conv(
                in_channels,
                kernel_size=1,
                out_channels=out_channels,
            ) if exists(out_channels) else nn.Identity()
        )
        
        self.main = nn.Sequential(
            Norm(num_groups, in_channels),
            Act(),
            Conv(
                in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=tuple(map(lambda k : (k - 1) // 2, kernel_size)),
            ),
            Down(
                out_channels,
                kernel_size,
                time_factor=time_factor,
                space_factor=space_factor,
                num_groups=num_groups,
            ) if exists(downsample) else nn.Identity(),
            Norm(num_groups, out_channels),
            Act(),
            Conv(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=tuple(map(lambda k : (k - 1) // 2, kernel_size)),
            ),
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
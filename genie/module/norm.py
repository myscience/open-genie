import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import group_norm

from einops import pack, rearrange, unpack

class AdaptiveGroupNorm(nn.Module):
    def __init__(
        self,
        dim_cond : int,
        num_groups: int,
        num_channels: int,
        cond_bias : bool = True,
        affine : bool = True,
        eps : float = 1e-5,
        device : str | None = None,
        dtype : str | None = None,
    ) -> None:
        super().__init__()
        
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias   = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.std = nn.Linear(dim_cond, self.num_channels)
        self.avg = nn.Linear(dim_cond, self.num_channels) if cond_bias else None
        
        self.reset_parameters()

    def reset_parameters(self) -> None:      
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        
        nn.init.ones_ (self.std.bias)
        nn.init.zeros_(self.std.weight)
        
        if self.avg is not None:
            nn.init.zeros_(self.avg.bias)
            nn.init.zeros_(self.avg.weight)
            
    def forward(self, inp : Tensor, cond : Tensor) -> Tensor:
        # Apply the standard group norm to the input.
        # Expected shape: [B, G, *]
        norm = group_norm(inp, self.num_groups, self.weight, self.bias, self.eps)
        norm, ps = pack([norm], 'b g *')
        
        # Condition is expected to have shape b d ...
        cond = rearrange(cond, 'b d ... -> b d (...)').mean(-1)
        
        # Rescale the normalized input to match the conditional statistics
        std = self.std(cond).unsqueeze(-1)
        avg = self.avg(cond).unsqueeze(-1) if self.avg is not None else 0
        
        out =  norm * std + avg
        return unpack(out, ps, 'b g *')[0]
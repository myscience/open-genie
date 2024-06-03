import torch.nn as nn
from torch import Tensor
from torch.nn.functional import group_norm

from einops import pack, unpack

class AdaptiveGroupNorm(nn.GroupNorm):
    def __init__(
        self,
        dim_cond : int,
        num_groups: int,
        num_channels: int,
        cond_bias : bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            num_groups=num_groups,
            num_channels=num_channels,
            **kwargs
        )
        
        self.std = nn.Linear(dim_cond, self.num_groups)
        self.avg = nn.Linear(dim_cond, self.num_groups) if cond_bias else None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        
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
        
        # Rescale the normalized input to match the conditional statistics
        std = self.std(cond).unsqueeze(-1)
        avg = self.avg(cond).unsqueeze(-1) if self.avg is not None else 0
        
        out =  norm * std + avg
        return unpack(out, ps)[0]
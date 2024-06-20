from itertools import pairwise
from uuid import uuid4
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple

from einops import rearrange
from collections import defaultdict

from genie.utils import default

class NamingProbe:
    
    def __init__(self, name_attr : str = 'name') -> None:
        super().__init__()
        
        self.depth = -1
        self.name_attr = name_attr
        
    def __call__(
        self,
        module : nn.Module,
        inp : Tuple[Tensor, ...],
        out : Tensor
    ) -> None:
        '''Custom torch hook designed to record hidden activations.
        NOTE: This function should be called (implicitly) by the
        forward hook registered on the desired module.
        '''
        
        self.depth += 1
        
        # Build unique module name as identifier
        name = f'{module._get_name().lower()}_{self.depth}_{uuid4().hex[:6]}'
        
        setattr(module, self.name_attr, name)

class RecordingProbe:
    
    def __init__(self) -> None:
        self._data : Dict[str, List[Tensor]] = defaultdict(list)
        
    @property
    def features(self) -> Dict[str, Tensor]:
        return {k: torch.cat(v) for k, v in self._data.items()}
    
    def __call__(
        self,
        module : nn.Module,
        inp : Tuple[Tensor, ...],
        out : Tensor
    ) -> None:
        '''Custom torch hook designed to record hidden activations.
        NOTE: This function should be called (implicitly) by the
        forward hook registered on the desired module.
        '''
        
        # Get the name of the module
        name = module.name if hasattr(module, 'name') else module._get_name().lower()
        
        feat = out.clone().detach()
        feat = rearrange(feat, 'b ... -> b (...)').contiguous()
        
        self._data[name].append(feat)
        
    def clean(self) -> None:
        '''Clear the recorded data.'''
        self._data.clear()
        
class ForwardBlock(nn.Module):
    
    def __init__(
        self,
        in_dim : int,
        out_dim : int | None = None,
        hid_dim : int | Tuple[int, ...] | None = 256,
        block  : nn.Module = nn.Linear,
        act_fn : nn.Module = nn.GELU,
        num_groups : int = 1,
        last_act : bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        
        out_dim = default(out_dim, in_dim)
        if isinstance(hid_dim, int): hid_dim = (hid_dim,)
        hid_dim = default(hid_dim, ())
        
        dims = (in_dim,) + hid_dim + (out_dim,)
        
        self.net = nn.Sequential(
            nn.GroupNorm(num_groups, in_dim),
            *[nn.Sequential(
                block(inp_dim, out_dim, **kwargs),
                act_fn() if l < len(dims) - 2 or last_act else nn.Identity()
            ) for l, (inp_dim, out_dim) in enumerate(pairwise(dims))],
        )
        
    def forward(
        self,
        inp : Tensor
    ) -> Tensor:
        return self.net(inp)
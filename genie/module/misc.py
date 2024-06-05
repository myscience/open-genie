import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple

from einops import rearrange
from collections import defaultdict

class RecordingProbe():
    
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
        name = module._get_name().lower()
        
        feat = out.clone().detach()
        feat = rearrange(feat, 'b ... -> b (...)').contiguous()
        
        self._data[name].append(feat)
        
    def clean(self) -> None:
        '''Clear the recorded data.'''
        self._data.clear()
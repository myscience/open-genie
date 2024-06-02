
import torch.nn as nn
from typing import Tuple

from lightning import LightningModule
from typing import Dict

from genie.module.video import CausalConv3d, ResidualBlock
from genie.module.video import SpaceUpsample

class VideoEncoder(nn.Module):
    '''Video Encoder as described in the paper:
    "Language Model Beats Diffusion - Tokenizer is Key
    to Visual Generation", Yu et al. (2024).

    This encoder employs a stack of causal convolutions
    to process the input video sequence.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

class VideoTokenizer(LightningModule):
    '''
    Video Tokenizer based on the MagViT-2 paper:
    "Language Model Beats Diffusion: Tokenizer is
    key to visual generation", Yu et al. (2024).
    '''
    
    def __init__(
        self,
        enc_desc : Tuple[str | Tuple[str, Dict], ...],
        dec_desc : Tuple[str | Tuple[str, Dict], ...],
        inp_dim : int = 128,
        out_dim : int | None = None,
        inp_channels: int = 3,
        kernel_size : int | Tuple[int, int, int] = 3,
        pad_mode : str = 'constant',
    ) -> None:
        super().__init__()
        
        inp_conv = CausalConv3d(
            inp_channels,
            inp_dim,
            kernel_size=kernel_size,
            pad_mode=pad_mode,
        )
        
        out_conv = CausalConv3d(
            inp_dim,
            inp_channels,
            kernel_size=kernel_size,
            pad_mode=pad_mode,
        )
        
        # Scan the blueprint to build the tokenizer
        self.enc_layers = nn.ModuleList([inp_conv])
        self.dec_layers = nn.ModuleList([out_conv])
        
        for layer in enc_desc:
            if isinstance(layer, str): layer = (layer, {})
            
            name, kwargs = layer
            match name:
                case 'residual':
                    self.enc_layers.append(ResidualBlock(**kwargs))
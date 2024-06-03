import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Tuple

from typing import Dict
from lightning import LightningModule

from itertools import zip_longest

from genie.module.quantization import LookupFreeQuantization
from genie.module.video import CausalConv3d, ResidualBlock
from genie.module.video import SpaceUpsample
from genie.utils import default, exists

class VideoEncoder(nn.Module):
    '''Video Encoder as described in the paper:
    "Language Model Beats Diffusion - Tokenizer is Key
    to Visual Generation", Yu et al. (2024).

    This encoder employs a stack of causal convolutions
    to process the input video sequence.
    '''
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
def get_module(name : str) -> nn.Module:
    match name:
        case 'residual':
            return ResidualBlock
        case 'causal':
            return CausalConv3d
        case 'space_upsample':
            return SpaceUpsample
        case _:
            raise ValueError(f'Unknown module name: {name}')

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
        # Lookup-Free Quantization parameters
        d_codebook : int = 18,
        n_codebook : int = 1,
        # lfq_input_dim : int | None = None,
        lfq_bias : bool = True,
        lfq_frac_sample : float = 1.,
        lfq_commit_weight : float = 0.25,
        lfq_entropy_weight : float = 0.1,
        lfq_diversity_weight : float = 1.,
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
        
        for enc_l, dec_l in zip_longest(enc_desc, dec_desc):
            if isinstance(enc_l, str): enc_l = (enc_l, {})
            if isinstance(dec_l, str): dec_l = (dec_l, {})
            
            name, kwargs = default(enc_l, (None, None))
            self.enc_layers.extend(
                [
                    get_module(name)(**kwargs)
                    for _ in range(kwargs.pop('n_rep', 1))
                    if exists(name) and exists(kwargs)
                ]
            )
            name, kwargs = default(dec_l, (None, None))
            self.dec_layers.extend(
                [
                    get_module(name)(**kwargs)
                    for _ in range(kwargs.pop('n_rep', 1))
                    if exists(name) and exists(kwargs)
                ]
            )
        
        # Check consistency between last encoder dimension, first
        # decoder dimension and the codebook dimension
        last_enc_dim = self.enc_layers[-1].out_channels
        first_dec_dim = self.dec_layers[0].in_channels
        assert last_enc_dim == first_dec_dim, 'Inconsistent encoder/decoder dimensions'
        assert last_enc_dim == d_codebook   , 'Codebook dimension mismatch with encoder/decoder'
        
        
        # Build the quantization module
        self.quant = LookupFreeQuantization(
            d_codebook       = d_codebook,
            n_codebook       = n_codebook,
            use_bias         = lfq_bias,
            frac_sample      = lfq_frac_sample,
            commit_weight    = lfq_commit_weight,
            entropy_weight   = lfq_entropy_weight,
            diversity_weight = lfq_diversity_weight,
        )
        
    def encode(
        self,
        video : Tensor,
    ) -> Tensor:
        enc_video = video
        
        for layer in self.enc_layers:
            enc_video = layer(enc_video)
            
        return enc_video
    
    def decode(
        self,
        quant_video : Tensor,
    ) -> Tensor:
        dec_video = quant_video
        for layer in self.dec_layers:
            dec_video = layer(dec_video)
            
        return dec_video
    
    @torch.no_grad()
    def tokenize(
        self,
        video : Tensor,
        beta : float = 100.,
        transpose : bool = False,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        
        enc_video = self.encode(video)
        (quant_video, idxs), _ = self.quant(
            enc_video,
            beta=beta,
            transpose=transpose
        )
        
        return quant_video, idxs
    
    def forward(
        self,
        video : Tensor,
        beta : float = 100.,
        transpose : bool = False,
    ) -> Tuple[Tensor, Tensor]:
        enc_video = self.encode(video)
        (quant_video, idxs), quant_loss = self.quant(enc_video, beta=beta, transpose=transpose)
        rec_video = self.decode(quant_video)
        
        # Compute the tokenizer loss
        raise NotImplementedError('Loss computation not implemented')
        
        return loss, (quant_loss,)
    
    # * Lightning core functions
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the training loss
        loss, *aux_losses = self(batch)
        
        # Log the training loss
        self.log_dict({
                'train_loss': loss,
                'train_quant_loss': aux_losses[0],
            }, on_step=True, on_epoch=True, sync_dist=True)
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the validation loss
        loss, *aux_losses = self(batch)
        
        # Log the training loss
        self.log_dict({
                'val_loss': loss,
                'val_quant_loss': aux_losses[0],
            }, on_step=True, on_epoch=True, sync_dist=True)
    
    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()
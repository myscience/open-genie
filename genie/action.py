from typing import Tuple
from torch import Tensor
import torch.nn as nn

from math import prod
from torch.nn.functional import mse_loss

from einops.layers.torch import Rearrange

from genie.module import parse_blueprint
from genie.module.quantization import LookupFreeQuantization
from genie.module.video import CausalConv3d, Downsample, Upsample
from genie.utils import Blueprint

REPR_ACT_ENC = (
    ('space-time_attn', {
        'n_repr' : 8,
        'n_heads': 8,
        'd_head': 64,
    }),
)

REPR_ACT_DEC = (
    ('space-time_attn', {
        'n_repr' : 8,
        'n_heads': 8,
        'd_head': 64,
    }),
)

class LatentAction(nn.Module):
    '''Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    '''
    
    def __init__(
        self,
        enc_desc: Blueprint,
        dec_desc: Blueprint,
        d_codebook: int,
        inp_channels: int = 3,
        inp_shape : int | Tuple[int, int] = (64, 64),
        ker_size : int | Tuple[int, int] = 3,
        n_embd: int = 256,
        n_codebook: int = 1,
        lfq_bias : bool = True,
        lfq_frac_sample : float = 1.,
        lfq_commit_weight : float = 0.25,
        lfq_entropy_weight : float = 0.1,
        lfq_diversity_weight : float = 1.,
        quant_loss_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        if isinstance(inp_shape, int): inp_shape = (inp_shape, inp_shape)
        
        self.proj_in = CausalConv3d(
            inp_channels,
            out_channels=n_embd,
            kernel_size=ker_size
        )
        
        self.proj_out = CausalConv3d(
            n_embd,
            out_channels=inp_channels,
            kernel_size=ker_size
        )
        
        # Build the encoder and decoder based on the blueprint
        self.enc_layers, self.enc_ext = parse_blueprint(enc_desc)
        self.dec_layers, self.dec_ext = parse_blueprint(dec_desc)
        
        # Keep track of space-time up/down factors
        enc_fact = prod(enc.factor for enc in self.enc_layers if isinstance(enc, (Downsample, Upsample)))
        dec_fact = prod(dec.factor for dec in self.dec_layers if isinstance(dec, (Downsample, Upsample)))
        
        assert enc_fact * dec_fact == 1, 'The product of the space-time up/down factors must be 1.'
        
        # Add the projections to the action space
        self.to_act = nn.Sequential(
                Rearrange('b c t ... -> b t (c ...)'),
                nn.Linear(
                    int(n_embd * enc_fact * prod(inp_shape)),
                    d_codebook,
                    bias=False,
                )
        )
        
        # Build the quantization module
        self.quant = LookupFreeQuantization(
            codebook_dim       = d_codebook,
            num_codebook       = n_codebook,
            use_bias         = lfq_bias,
            frac_sample      = lfq_frac_sample,
            commit_weight    = lfq_commit_weight,
            entropy_weight   = lfq_entropy_weight,
            diversity_weight = lfq_diversity_weight,
        )
        
        self.d_codebook = d_codebook
        self.n_codebook = n_codebook
        self.quant_loss_weight = quant_loss_weight
        
    def sample(self, idxs : Tensor) -> Tensor:
        '''Sample the action codebook values based on the indices.'''
        return self.quant.codebook[idxs]
        
    def encode(
        self,
        video: Tensor,
        mask : Tensor | None = None,
        transpose : bool = False,
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor]:
        video = self.proj_in(video)
        
        # Encode the video frames into latent actions
        for enc in self.enc_layers:
            video = enc(video, mask=mask)
        
        # Project to latent action space
        act : Tensor = self.to_act(video)

        # Quantize the latent actions
        (act, idxs), q_loss = self.quant(act, transpose=transpose)
        
        return (act, idxs, video), q_loss
    
    def decode(
        self,
        video : Tensor,
        q_act : Tensor,
    ) -> Tensor:        
        # Decode the video frames based on past history and
        # the quantized latent actions
        for dec, has_ext in zip(self.dec_layers, self.dec_ext):
            video = dec(
                video,
                cond=(
                    None, # No space condition
                    q_act if has_ext else None,
                )
            )
            
        recon = self.proj_out(video)
        
        return recon
        
    def forward(
        self,
        video: Tensor,
        mask : Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        
        # Encode the video frames into latent actions
        (act, idxs, enc_video), q_loss = self.encode(video, mask=mask)
        
        # Decode the last video frame based on all the previous
        # frames and the quantized latent actions
        recon = self.decode(enc_video, act)
        
        # Compute the reconstruction loss
        # Reconstruction loss
        rec_loss = mse_loss(recon, video)
        
        # Compute the total loss by combining the individual
        # losses, weighted by the corresponding loss weights
        loss = rec_loss\
            + q_loss * self.quant_loss_weight
        
        return idxs, loss, (
            rec_loss,
            q_loss,
        )
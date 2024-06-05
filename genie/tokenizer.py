import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim import Optimizer
from torchvision.models import get_model
from torch.nn.functional import mse_loss

from typing import Any, Tuple
from typing import Dict, Callable, Iterable
from itertools import zip_longest

from lightning import LightningModule

from genie.module.loss import GANLoss
from genie.module.loss import PerceptualLoss
from genie.module.quantization import LookupFreeQuantization
from genie.module.video import CausalConv3d, ResidualBlock
from genie.module.video import SpaceUpsample
from genie.utils import default, exists

OptimizerCallable = Callable[[Iterable], Optimizer]
        
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
    
    This tokenizer employs a stack of causal
    convolutions to process the input video sequence.
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
        # Misc parameters
        optimizer : OptimizerCallable = AdamW,
        perceptual_model : str = 'vgg16',
        perc_feat_layers : str | Iterable[str] = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
        gan_loss_weight : float = 1.,
        perc_loss_weight : float = 1.,
        quant_loss_weight : float = 1.,
        num_frames_per_batch : int = 4,
    ) -> None:
        super().__init__()
        
        self.optimizer = optimizer
        
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
            self.enc_ext = kwargs.pop('has_ext', False)
            self.enc_layers.extend(
                [
                    get_module(name)(**kwargs)
                    for _ in range(kwargs.pop('n_rep', 1))
                    if exists(name) and exists(kwargs)
                ]
            )
            name, kwargs = default(dec_l, (None, None))
            self.dec_ext = kwargs.pop('has_ext', False)
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
        
        # If the perceptual loss is enabled, load the perceptual model
        self.perc_crit = PerceptualLoss(
                model_name=perceptual_model,
                feat_layers=perc_feat_layers,
                num_frames=num_frames_per_batch,
            ) if perc_loss_weight > 0 else nn.Identity()
        
        # If the GAN loss is enabled, load the Discriminator model
        self.gan_crit = GANLoss(
                num_frames=num_frames_per_batch,
            ) if gan_loss_weight > 0 else nn.Identity()
        
        
        self.gan_loss_weight  = gan_loss_weight
        self.perc_loss_weight = perc_loss_weight
        self.quant_loss_weight = quant_loss_weight
        self.save_hyperparameters()
        
    def encode(
        self,
        video : Tensor,
        cond : Tensor | None = None,
    ) -> Tensor:
        enc_video = video
        
        for layer, has_ext in self.enc_layers:
            enc_video = layer(enc_video, cond) if has_ext else layer(enc_video)
            
        return enc_video
    
    def decode(
        self,
        quant : Tensor,
        cond : Tensor | None = None,
    ) -> Tensor:
        cond = default(cond, quant)
        
        rec_video = quant
        for layer, has_ext in zip(self.dec_layers, self.dec_ext):
            rec_video = layer(rec_video, cond) if has_ext else layer(rec_video)
            
        return rec_video
    
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
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        enc_video = self.encode(video)
        (quant_video, idxs), quant_loss = self.quant(enc_video, beta=beta, transpose=transpose)
        rec_video = self.decode(quant_video)
        
        # * Compute the tokenizer loss
        # Reconstruction loss
        rec_loss = mse_loss(rec_video, video)
        
        # GAN loss (if available)
        gan_loss = self.gan_crit(rec_video, video)
        
        # Perceptual loss (if available)
        perc_loss = self.perc_crit(rec_video, video)
        
        # Compute the total loss by combining the individual
        # losses, weighted by the corresponding loss weights
        loss = rec_loss\
            + gan_loss   * self.gan_loss_weight\
            + perc_loss  * self.perc_loss_weight\
            + quant_loss * self.quant_loss_weight\
        
        return loss, (
            rec_loss,
            gan_loss if self.gan_loss_weight > 0 else None,
            perc_loss if self.perc_loss_weight > 0 else None,
            quant_loss
        )
    
    # * Lightning core functions
    
    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the training loss
        loss, aux_losses = self(batch)
        
        # Log the training loss
        self.log_dict(
            {
                'train_loss': loss,
                'train_rec_loss'  : aux_losses[0],
                'train_gan_loss'  : aux_losses[1],
                'train_perc_loss' : aux_losses[2],
                'train_quant_loss': aux_losses[3],
            },
            logger=True,
            on_step=True,
            sync_dist=True
        )
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the validation loss
        loss, aux_losses = self(batch)
        
        # Log the training loss
        self.log_dict(
            {
                'val_loss': loss,
                'val_rec_loss'  : aux_losses[0],
                'val_gan_loss'  : aux_losses[1],
                'val_perc_loss' : aux_losses[2],
                'val_quant_loss': aux_losses[3],
            },
            on_step=True,
            logger=True,
            sync_dist=True
        )
        
        return loss
    
    def on_validation_end(self) -> None:
        # Maybe put here example of video reconstructions?
        pass
    
    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )
        
        return optim
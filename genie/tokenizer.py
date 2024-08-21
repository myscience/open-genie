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
from genie.utils import Blueprint, default, exists

from genie.module import parse_blueprint

OptimizerCallable = Callable[[Iterable], Optimizer]

MAGVIT2_ENC_DESC = (
    ('causal-conv3d', {
        'in_channels': 3,
        'out_channels': 128,
        'kernel_size': 3,
    }),
    ('video-residual', {
        'n_rep': 4,
        'in_channels': 128,
    }),
    ('spacetime_downsample', {
        'in_channels': 128,
        'out_channels': 128,
        'kernel_size': 3,
        'time_factor': 1,
        'space_factor': 2,
    }),
    ('video-residual', {
        'in_channels': 128,
        'out_channels': 256,
    }),
    ('video-residual', {
        'n_rep': 3,
        'in_channels': 256,
    }),
    ('spacetime_downsample', {
        'in_channels': 256,
        'out_channels': 256,
        'kernel_size': 3,
        'time_factor': 2,
        'space_factor': 2,
    }),
    ('video-residual', {
        'n_rep': 4,
        'in_channels': 256,
    }),
    ('spacetime_downsample', {
        'in_channels': 256,
        'out_channels': 256,
        'kernel_size': 3,
        'time_factor': 2,
        'space_factor': 2,
    }),
    ('video-residual', {
        'in_channels': 256,
        'out_channels': 512,
    }),
    ('video-residual', {
        'n_rep': 7,
        'in_channels': 512,
    }),
    ('group_norm', {
        'num_groups': 8,
        'num_channels': 512,
    }),
    ('silu', {}),
    ('causal-conv3d', {
        'in_channels': 512,
        'out_channels': 18,
        'kernel_size': 1,
    })
)

MAGVIT2_DEC_DESC = (
    ('causal-conv3d', {
        'in_channels': 18,
        'out_channels': 512,
        'kernel_size': 3,
    }),
    ('video-residual', {
        'n_rep': 4,
        'in_channels': 512,
    }),
    ('adaptive_group_norm', {
        'dim_cond' : 18,
        'num_groups': 8,
        'num_channels': 512,
        'has_ext' : True,
    }),
    ('video-residual', {
        'n_rep': 4,
        'in_channels': 512,
    }),
    ('depth2spacetime_upsample', {
        'in_channels': 512,
        'kernel_size': 3,
        'time_factor': 2,
        'space_factor': 2,
    }),
    ('adaptive_group_norm', {
        'dim_cond' : 18,
        'num_groups': 8,
        'num_channels': 512,
        'has_ext' : True,
    }),
    ('video-residual', {
        'in_channels': 512,
        'out_channels': 256,
    }),
    ('video-residual', {
        'n_rep': 3,
        'in_channels': 256,
    }),
    ('depth2spacetime_upsample', {
        'in_channels': 256,
        'kernel_size': 3,
        'time_factor': 2,
        'space_factor': 2,
    }),
    ('adaptive_group_norm', {
        'dim_cond' : 18,
        'num_groups': 8,
        'num_channels': 256,
        'has_ext' : True,
    }),
    ('video-residual', {
        'n_rep' : 4,
        'in_channels': 256,
    }),
    ('depth2spacetime_upsample', {
        'in_channels': 256,
        'kernel_size': 3,
        'time_factor': 1,
        'space_factor': 2,
    }),
    ('adaptive_group_norm', {
        'dim_cond' : 18,
        'num_groups': 8,
        'num_channels': 256,
        'has_ext' : True,
    }),
    ('video-residual', {
        'in_channels': 256,
        'out_channels': 128,
    }),
    ('video-residual', {
        'n_rep' : 3,
        'in_channels': 128,
    }),
    ('group_norm', {
        'num_groups': 8,
        'num_channels': 128,
    }),
    ('silu', {}),
    ('causal-conv3d', {
        'in_channels': 128,
        'out_channels': 3,
        'kernel_size': 3,
    })
)

REPR_TOK_ENC = (
    ('spacetime_downsample', {
        'in_channels' : 3,
        'kernel_size' : 3,
        'out_channels' : 512,
        'time_factor' : 1,
        'space_factor' : 4,
    }),
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 64,
        'transpose' : True,
    }),
)

REPR_TOK_DEC = (
    ('space-time_attn', {
        'n_rep' : 8,
        'n_head': 8,
        'd_head': 64,
        'transpose' : True,
    }),
    ('depth2spacetime_upsample', {
        'in_channels' : 512,
        'kernel_size' : 3,
        'out_channels' : 3,
        'time_factor' : 1,
        'space_factor' : 4,
    })
)

def get_enc(name : str) -> Blueprint:
    match name:
        case 'magvit2':
            return MAGVIT2_ENC_DESC
        case 'repr_tok':
            return REPR_TOK_ENC
        case _:
            raise ValueError(f'Unknown encoder: {name}')

def get_dec(name : str) -> Blueprint:
    match name:
        case 'magvit2':
            return MAGVIT2_DEC_DESC
        case 'repr_tok':
            return REPR_TOK_DEC
        case _:
            raise ValueError(f'Unknown decoder: {name}')

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
        enc_desc : Blueprint,
        dec_desc : Blueprint,
        disc_kwargs : Dict[str, Any] = {},
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
        perc_feat_layers : str | Iterable[str] = ('features.6', 'features.13', 'features.18', 'features.25'),
        gan_discriminate : str = 'frames',
        gan_frames_per_batch : int = 4,
        gan_loss_weight : float = 1.,
        perc_loss_weight : float = 1.,
        quant_loss_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        self.optimizer = optimizer
        
        # Scan the blueprint to build the tokenizer        
        self.enc_layers, self.enc_ext = parse_blueprint(enc_desc)
        self.dec_layers, self.dec_ext = parse_blueprint(dec_desc)
        
        # Check consistency between last encoder dimension, first
        # decoder dimension and the codebook dimension
        # last_enc_dim = list(self.enc_layers.modules())[-1].out_channels
        last_enc_dim = [m.out_channels for m in self.enc_layers.modules() if hasattr(m, 'out_channels')][-1]
        first_dec_dim = self.dec_layers[0].in_channels
        assert last_enc_dim == first_dec_dim, 'Inconsistent encoder/decoder dimensions'
        # assert last_enc_dim == d_codebook   , 'Codebook dimension mismatch with encoder/decoder'
        
        # Build the quantization module
        self.quant = LookupFreeQuantization(
            codebook_dim     = d_codebook,
            num_codebook     = n_codebook,
            input_dim        = last_enc_dim,
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
                num_frames=gan_frames_per_batch,
            ) if perc_loss_weight > 0 else nn.Identity()
        
        # If the GAN loss is enabled, load the Discriminator model
        self.gan_crit = GANLoss(
                discriminate=gan_discriminate,
                num_frames=gan_frames_per_batch,
                **disc_kwargs,
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
        
        for layer, has_ext in zip(self.enc_layers, self.enc_ext):
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
        transpose : bool = True,
    ) -> Tuple[Tensor, Tensor]:
        self.eval()
        
        enc_video = self.encode(video)
        (quant_video, idxs), _ = self.quant(
            enc_video,
            beta=beta,
            transpose=transpose
        )
        
        self.train()
        
        return quant_video, idxs
    
    def forward(
        self,
        video : Tensor,
        beta : float = 100.,
        transpose : bool = True,
    ) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        enc_video = self.encode(video)
        (quant_video, idxs), quant_loss = self.quant(enc_video, beta=beta, transpose=transpose)
        rec_video = self.decode(quant_video)
        
        # * Compute the tokenizer loss
        # Reconstruction loss
        rec_loss = mse_loss(rec_video, video)
        
        # GAN loss (if available)
        gen_loss = self.gan_crit(rec_video, video, train_gen=True)
        dis_loss = self.gan_crit(rec_video, video, train_gen=False)
        
        # Perceptual loss (if available)
        perc_loss = self.perc_crit(rec_video, video)
        
        # Compute the total loss by combining the individual
        # losses, weighted by the corresponding loss weights
        loss = rec_loss\
            + gen_loss   * self.gan_loss_weight\
            + dis_loss   * self.gan_loss_weight\
            + perc_loss  * self.perc_loss_weight\
            + (quant_loss * self.quant_loss_weight) if exists(quant_loss) else 0\
        
        return loss, (
            rec_loss,
            gen_loss if self.gan_loss_weight > 0 else 0,
            dis_loss if self.gan_loss_weight > 0 else 0,
            perc_loss if self.perc_loss_weight > 0 else 0,
            quant_loss if exists(quant_loss) and self.quant_loss_weight > 0 else 0,
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
                'train_gen_loss'  : aux_losses[1],
                'train_dis_loss'  : aux_losses[2],
                'train_perc_loss' : aux_losses[3],
                'train_quant_loss': aux_losses[4],
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
                'val_gen_loss'  : aux_losses[1],
                'val_dis_loss'  : aux_losses[2],
                'val_perc_loss' : aux_losses[3],
                'val_quant_loss': aux_losses[4],
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
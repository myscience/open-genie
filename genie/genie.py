from einops import rearrange
from lightning import LightningModule
from torch import Tensor
import torch
from torch.optim import AdamW
from torch.optim import Optimizer

from genie.action import LatentAction
from genie.dynamics import DynamicsModel
from genie.tokenizer import VideoTokenizer

from typing import Callable, Iterable

from genie.utils import default

OptimizerCallable = Callable[[Iterable], Optimizer]

class Genie(LightningModule):
    '''
    Generative Interactive Environment model from Bruce et al. (2024).
    The model is composed of:
    - A (pre-trained) video tokenizer based on the MaskVit-2 architecture.
    - A Latent Action model that build a (quantized) dictionary of latent actions
    - A Dynamics Model that predicts the next frame given the current frame and the latent action.
    '''
    def __init__(
        self,
        tokenizer : VideoTokenizer,
        optimizer : OptimizerCallable = AdamW,
        img_prompt : Tensor | None = None,
    ):
        super().__init__()
        
        # Pre-trained video tokenizer
        self.tokenizer = tokenizer
        
        self.latent_action = LatentAction(
            self.enc_desc,
            self.dec_desc,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            inp_shape=self.inp_shape,
            ker_size=self.ker_size,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            lfq_bias=self.lfq_bias,
            lfq_frac_sample=self.lfq_frac_sample,
            lfq_commit_weight=self.lfq_commit_weight,
            lfq_entropy_weight=self.lfq_entropy_weight,
            lfq_diversity_weight=self.lfq_diversity_weight,
        )
        
        self.dynamics_model = DynamicsModel(
            desc=TEST_DESC,
            tok_vocab=self.tok_codebook,
            act_vocab=self.act_codebook,
            embed_dim=self.embed_dim,
        )
        
        self.optimizer = optimizer
        self.img_prompt = img_prompt
        
        self.save_hyperparameters()

    @torch.no_grad()
    def forward(
        self,
        prompt : Tensor,
        actions : Tensor,
        num_frames : int | None = None,
        steps_per_frame : int = 25,
    ) -> Tensor:
        '''
        Inference mode for the model. Generate videos from an initial
        image prompt and a sequence of latent actions.
        '''
        num_frames = default(num_frames, actions.shape[1])
        
        # Make sure prompt has correct shape for video
        match prompt.dim():
            case 3: pattern = 'b h w -> b 1 1 h w'
            case 4: pattern = 'b c h w -> b c 1 h w'
            case 5: pattern = 'b c t h w -> b c t h w'
            case _: raise ValueError('Prompt must have 3, 4 or 5 dimensions')
        
        prompt = rearrange(prompt, pattern)
        
        # Tokenize the input prompt
        tokens = self.tokenizer.tokenize(prompt)
        
        for t in range(num_frames):
            # Predict the next frame based on the previous frame and the action
            new_tok = self.dynamics_model.generate(
                tokens,
                actions[:, :t],
                steps=steps_per_frame,
            )
            
            # Add the new frame to the video
            tokens = torch.stack((tokens, new_tok), dim=2)
            
        # Return the generated video
        video = self.tokenizer.decode(tokens)
        
        return video
    
    def compute_loss(self, video : Tensor) -> Tensor:
        # Tokenize the input video
        tokens = self.tokenizer.tokenize(video)
        
        # Extract latent actions from the video
        act_id, act_loss, (act_rec_loss, act_q_loss) = self.latent_action(video)
        
        # Compute the next-frame prediction loss via the dynamics model 
        dyn_loss = self.dynamics_model.compute_loss(tokens, act_id)
        
        # Combine both latent action and dynamics model losses
        loss = act_loss + dyn_loss
        
        return loss, (
            ('act_loss', act_loss),
            ('dyn_loss', dyn_loss),
            ('act_rec_loss', act_rec_loss),
            ('act_q_loss', act_q_loss),
        )

    def training_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the training loss
        loss, aux_losses = self.compute_loss(batch)
        
        # Log the training loss
        self.log_dict(
            {**{'train_loss' : loss}, **{f'train/{k}': v for k, v in aux_losses}},
            logger=True,
            on_step=True,
            sync_dist=True,
        )
        
        return loss
    
    def validation_step(self, batch : Tensor, batch_idx : int) -> Tensor:
        # Compute the validation loss
        loss, aux_losses = self.compute_loss(batch)
        
        # Log the training loss
        self.log_dict(
            {**{'val_loss' : loss}, **{f'val/{k}': v for k, v in aux_losses}},
            logger=True,
            on_step=True,
            sync_dist=True,
        )
        
        return loss
    
    def on_validation_end(self) -> None:
        '''Generate sample videos at the end of the validation loop'''
        
        # Generate a sample video from a given image prompt and random actions
        num_frames = 16
        prompt = default(self.img_prompt, torch.randn(1, 3, 64, 64))
        actions = torch.randint(0, self.latent_action.d_codebook, size=(num_frames,))
        
        video = self(
            prompt,
            actions,
            num_frames=num_frames,
            steps_per_frame=25
        )
        
        self.logger.experiment.add_video(
            f'Generated Video #1',
            video,
            global_step=self.global_step,
        )

    def configure_optimizers(self) -> Optimizer:
        optim = self.optimizer(
            self.parameters(),
        )
        
        return optim
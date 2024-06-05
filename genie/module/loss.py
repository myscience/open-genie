import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import get_model

from torch.nn.functional import relu
from torch.nn.functional import mse_loss

from typing import Iterable

from genie.module.misc import RecordingProbe
from genie.module.image import FrameDiscriminator
from genie.utils import pick_frames

class PerceptualLoss(nn.Module):
    
    def __init__(
        self,
        model_name : str = 'vgg16',
        model_weights : str | None = 'DEFAULT',
        num_frames : int = 4,
        feat_layers : str | Iterable[str] = ('relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'),
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        self.percept_model = get_model(model_name, weights=model_weights)
            
        # Freeze the perceptual model
        self.percept_model.eval()
        for param in self.percept_model.parameters():
            param.requires_grad = False
        
        # Attach hooks to the model at desired locations
        self.probe = RecordingProbe()
        self.hook_handles = [
            module.register_forward_hook(self.probe)
            for name, module in self.percept_model.named_modules()
            if name in feat_layers
        ]
        
    def forward(self, rec_video : Tensor, inp_video : Tensor) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        fake_frames = pick_frames(rec_video, frame_idxs)
        real_frames = pick_frames(inp_video, frame_idxs)
        
        frame_idxs = torch.cat([
            torch.randint(0, t, self.num_frames, device=inp_video.device)
            for _ in range(b)]
        )
        
        # Get the perceptual features for the input
        _ = self.percept_model(fake_frames)
        fake_feat = self.probe.features
        self.probe.clean()
        
        # Get the perceptual features for the target
        _ = self.percept_model(real_frames)
        real_feat = self.probe.features
        self.probe.clean()
        
        # Perceptual loss is the average MSE between the features
        return torch.stack([
            mse_loss(fake_feat[k], real_feat[k])
            for k in fake_feat.keys()
        ]).mean()
        
class GANLoss(nn.Module):
    
    def __init__(
        self,
        num_frames : int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.disc = FrameDiscriminator(**kwargs)
        
        self.num_frames = num_frames
        
    def forward(
        self,
        rec_video : Tensor,
        inp_video : Tensor,
        train_gen : bool, 
    ) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        frame_idxs = torch.cat([
            torch.randint(0, t, self.num_frames, device=inp_video.device)
            for _ in range(b)]
        )
        fake_frames = pick_frames(rec_video, frame_idxs)
        real_frames = pick_frames(inp_video, frame_idxs)
        
        # Compute discriminator opinions for real and fake frames
        fake_score : Tensor = self.disc(fake_frames) if     train_gen else self.disc(fake_frames.detach())
        real_score : Tensor = self.disc(real_frames) if not train_gen else None
        
        # Compute hinge loss for the discriminator
        gan_loss = -fake_score.mean() if train_gen else (relu(1 + fake_score) + relu(1 - real_score)).mean()
        
        return gan_loss
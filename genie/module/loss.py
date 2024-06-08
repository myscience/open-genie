import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import get_model

from torch.nn.functional import relu
from torch.nn.functional import mse_loss

from typing import Iterable, Tuple

from genie.module.misc import NamingProbe
from genie.module.misc import RecordingProbe
from genie.module.discriminator import FrameDiscriminator, VideoDiscriminator
from genie.utils import pick_frames

VGG16_RELU_LAYERS = [
    'features.1',
    'features.3',
    'features.6',
    'features.8',
    'features.11',
    'features.13',
    'features.15',
    'features.18',
    'features.20',
    'features.22',
    'features.25',
    'features.27',
    'features.29',
    'classifier.1',
    'classifier.4',
]

class PerceptualLoss(nn.Module):
    
    def __init__(
        self,
        model_name : str = 'vgg16',
        model_weights : str | None = 'DEFAULT',
        num_frames : int = 4,
        feat_layers : str | Iterable[str] = ('features.6', 'features.13', 'features.18', 'features.25'),
    ) -> None:
        super().__init__()
        
        self.num_frames = num_frames
        self.percept_model = get_model(model_name, weights=model_weights)
            
        # Freeze the perceptual model
        self.percept_model.eval()
        for param in self.percept_model.parameters():
            param.requires_grad = False
            
        # Attach the naming probe to make sure every layer
        # in the percept model has a unique identifier
        self.namer = NamingProbe()
        handles = [
            module.register_forward_hook(self.namer)
            for name, module in self.percept_model.named_modules()
        ]
        
        # Fake forward pass to the model to trigger the probe
        with torch.no_grad():
            _ = self.percept_model(torch.randn(1, 3, 224, 224))
        for handle in handles: handle.remove()
        
        # Attach hooks to the model at desired locations
        self.probe = RecordingProbe()
        self.hook_handles = [
            module.register_forward_hook(self.probe)
            for name, module in self.percept_model.named_modules()
            if name in feat_layers
        ]
        
        assert len(self.hook_handles) > 0, 'No valid layers found in the perceptual model.'
        
    def forward(self, rec_video : Tensor, inp_video : Tensor) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        
        frames_idxs = torch.cat([
            torch.randperm(t, device=inp_video.device)[:self.num_frames]
            for _ in range(b)]
        )
        
        fake_frames = pick_frames(rec_video, frames_idxs=frames_idxs)
        real_frames = pick_frames(inp_video, frames_idxs=frames_idxs)
        
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
        
    def __del__(self) -> None:
        for handle in self.hook_handles:
            handle.remove()

class GANLoss(nn.Module):
    
    def __init__(
        self,
        discriminate : str = 'frames',
        num_frames : int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        
        assert discriminate in ('frames', 'video'), 'Invalid discriminator type. Must be either "frames" or "video".'
        
        self.disc = FrameDiscriminator(**kwargs) if discriminate == 'frames' else VideoDiscriminator(**kwargs)
        
        self.num_frames = num_frames
        self.discriminate = discriminate
        
    def get_examples(
        self,
        rec_video : Tensor,
        inp_video : Tensor,
    ) -> Tuple[Tensor, Tensor]:
        b, c, t, h, w = inp_video.shape
        
        if self.discriminate == 'video':
            return rec_video, inp_video
        
        # Extract a set of random frames from the input video
        frame_idxs = torch.cat([
            torch.randperm(t, device=inp_video.device)[:self.num_frames]
            for _ in range(b)]
        )
        fake = pick_frames(rec_video, frame_idxs)
        real = pick_frames(inp_video, frame_idxs)
        
        return fake, real
        
    def forward(
        self,
        rec_video : Tensor,
        inp_video : Tensor,
        train_gen : bool, 
    ) -> Tensor:
        b, c, t, h, w = inp_video.shape
        
        # Extract a set of random frames from the input video
        fake, real = self.get_examples(rec_video, inp_video)
        
        # Compute discriminator opinions for real and fake frames
        fake_score : Tensor = self.disc(fake) if     train_gen else self.disc(fake.detach())
        real_score : Tensor = self.disc(real) if not train_gen else None
        
        # Compute hinge loss for the discriminator
        gan_loss = -fake_score.mean() if train_gen else (relu(1 + fake_score) + relu(1 - real_score)).mean()
        
        return gan_loss
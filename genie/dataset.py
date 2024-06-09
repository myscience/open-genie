from pathlib import Path

from torchvision.datasets import Kinetics

from typing import Callable, Tuple

from genie.module.data import LightningDataset

class LightningKinetics(LightningDataset):
    '''Lightning Dataset class for the Tiny Stories dataset. The Tiny
    Stories dataset is a small dataset of short stories, each consisting
    of a few sentences. The dataset is used for training a language model.
    '''
    
    def __init__(
        self,
        root: str | Path,
        frames_per_clip: int,
        num_classes: str = '400',
        frame_rate: int | None = None,
        step_between_clips: int = 1,
        transform: Callable | None = None,
        extensions: Tuple[str, ...] = ('avi', 'mp4'),
        download: bool = False,
        num_download_workers: int = 1,
        num_workers: int = 1,
        output_format: str = 'CTHW',
        **kwargs,    
    ) -> None:
        super().__init__(**kwargs)
        
        self.root = root
        
        self.download = download
        self.transform = transform
        self.extensions = extensions
        self.frame_rate = frame_rate
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.output_format = output_format
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_download_workers = num_download_workers
        
        self.save_hyperparameters()
        
    def setup(self, stage: str) -> None:

        match stage:
            case 'fit':
                self.train_dataset = Kinetics(
                    root=self.root,
                    split = 'train',
                    download             = self.download,
                    transform            = self.transform,
                    extensions           = self.extensions,
                    frame_rate           = self.frame_rate,
                    num_classes          = self.num_classes,
                    num_workers          = self.num_workers,
                    output_format        = self.output_format,
                    frames_per_clip      = self.frames_per_clip,
                    step_between_clips   = self.step_between_clips,
                    num_download_workers = self.num_download_workers,
                )
                self.valid_dataset = Kinetics(
                    root=self.root,
                    split = 'val',
                    download             = self.download,
                    transform            = self.transform,
                    extensions           = self.extensions,
                    frame_rate           = self.frame_rate,
                    num_classes          = self.num_classes,
                    num_workers          = self.num_workers,
                    output_format        = self.output_format,
                    frames_per_clip      = self.frames_per_clip,
                    step_between_clips   = self.step_between_clips,
                    num_download_workers = self.num_download_workers,
                )
            case 'test':
                self.test__dataset = Kinetics(
                    root=self.root,
                    split = 'test',
                    download             = self.download,
                    transform            = self.transform,
                    extensions           = self.extensions,
                    frame_rate           = self.frame_rate,
                    num_classes          = self.num_classes,
                    num_workers          = self.num_workers,
                    output_format        = self.output_format,
                    frames_per_clip      = self.frames_per_clip,
                    step_between_clips   = self.step_between_clips,
                    num_download_workers = self.num_download_workers,
                )
            case _:
                raise ValueError(f'Invalid stage: {stage}')
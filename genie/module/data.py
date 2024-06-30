from einops import rearrange
import yaml
import torch
from cv2 import VideoCapture
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from cv2 import CAP_PROP_POS_FRAMES
from cv2 import CAP_PROP_FRAME_COUNT

from os import listdir, path
from abc import abstractmethod
from random import randint

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from lightning import LightningDataModule

from typing import Callable

from genie.utils import default, exists
from genie.utils import default_iterdata_worker_init

class LightningDataset(LightningDataModule):
    '''
        Abstract Lightning Data Module that represents a dataset we
        can train a Lightning module on.
    '''
    
    @classmethod
    def from_config(cls, conf_path : str, *args, key : str = 'dataset') -> 'LightningDataset':
        '''
        Construct a Lightning DataModule from a configuration file.
        '''

        with open(conf_path, 'r') as f:
            conf = yaml.safe_load(f)

        data_conf = conf[key]

        return cls(
            *args,
            **data_conf,
        )

    def __init__(
        self,
        *args,
        batch_size : int = 16,
        num_workers : int = 0,
        train_shuffle : bool | None = None,
        val_shuffle   : bool | None = None,
        val_batch_size : None | int = None,
        worker_init_fn : None | Callable = None,
        collate_fn     : None | Callable = None,
        train_sampler  : None | Callable = None, 
        val_sampler    : None | Callable = None,
        test_sampler   : None | Callable = None, 
    ) -> None:
        super().__init__()

        self.train_dataset = None
        self.valid_dataset = None
        self.test__dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers    = num_workers
        self.batch_size     = batch_size
        self.train_shuffle  = train_shuffle
        self.val_shuffle    = val_shuffle
        self.train_sampler  = train_sampler
        self.valid_sampler  = val_sampler
        self.test__sampler  = test_sampler
        self.collate_fn     = collate_fn
        self.worker_init_fn = worker_init_fn
        self.val_batch_size = val_batch_size

    @abstractmethod
    def setup(self, stage: str) -> None:
        msg = \
        '''
        This is an abstract datamodule class. You should use one of
        the concrete subclasses that represents an actual dataset.
        '''

        raise NotImplementedError(msg)

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_iterdata_worker_init)
        else:
            worker_init_fn = self.worker_init_fn
        
        return DataLoader(
            self.train_dataset,                  # type: ignore
            sampler        = self.train_sampler, # type: ignore
            batch_size     = self.batch_size,
            shuffle        = self.train_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_iterdata_worker_init)
        else:
            worker_init_fn = self.worker_init_fn
            
        return DataLoader(
            self.valid_dataset,                  # type: ignore
            sampler        = self.valid_sampler, # type: ignore
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_iterdata_worker_init)
        else:
            worker_init_fn = self.worker_init_fn
            
        return DataLoader(
            self.test__dataset,                  # type: ignore
            sampler        = self.test__sampler, # type: ignore
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = worker_init_fn,
        )
        
class Platformer2D(Dataset):
    
    def __init__(
        self,
        root : str,
        split : str = 'train',
        env_name : str = 'Coinrun',
        padding : str = 'none',
        randomize : bool = False,
        transform : Callable | None = None,
        num_frames : int = 16,
        output_format: str = 't c h w',
    ) -> None:
        super().__init__()
        
        self.root = path.join(root, env_name, split)
        self.split = split
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.output_format = output_format
        self.transform = transform if exists(transform) else lambda x: x
        
        # Get all the file path based on the split
        self.file_names = [
            path.join(self.root, f)
            for f in listdir(self.root)
        ]
        
    def __len__(self) -> int:
        return len(self.file_names)
    
    def __getitem__(self, idx: int) -> Tensor:
        video_path = self.file_names[idx]
        
        video = self.load_video_slice(
            video_path,
            self.num_frames,
            None if self.randomize else 0,
        )
        
        return video
    
    def load_video_slice(
        self,
        video_path : str,
        num_frames : int,
        start_frame : int | None = None
    ) -> Tensor:
        cap = VideoCapture(video_path)
        total_frames = int(cap.get(CAP_PROP_FRAME_COUNT))
        
        # If video is shorted than the requested number of frames
        # we just return the whole video
        num_frames = min(num_frames, total_frames)
        
        start_frame = start_frame if exists(start_frame) else randint(0, total_frames - num_frames)
        cap.set(CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if ret:
                # *Frame was successfully read, parse it
                frame = cvtColor(frame, COLOR_BGR2RGB) 
                frame = torch.from_numpy(frame)
                frames.append(frame)
                
            else:
                # * We reached the end of video
                # Deal with padding and return
                match self.padding:
                    case 'none': pass
                    case 'repeat':
                        frames.extend([frames[-1]] * (num_frames - len(frames)))
                    case 'zero':
                        frames.extend([
                                torch.zeros_like(frames[-1])
                            ] * (num_frames - len(frames))
                        )
                    case 'random':
                        frames.extend([
                            torch.rand_like(frames[-1])
                            ] * (num_frames - len(frames))
                        )
                    case _:
                        raise ValueError(f'Invalid padding type: {self.padding}')
                break

        cap.release()
        video = torch.stack(frames) / 255.
        video = rearrange(video, f't h w c -> {self.output_format}')
        
        video = self.transform(video)
        
        return video
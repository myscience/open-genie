import torch

from einops import rearrange

from torch import Tensor
from torch.utils.data import get_worker_info

from typing import TypeVar, Tuple

T = TypeVar('T')
D = TypeVar('D')

Blueprint = Tuple[str | Tuple[str, dict], ...]

def exists(var : T | None) -> bool:
    return var is not None

def default(var : T | None, val : D) -> T | D:
    return var if exists(var) else val

def enlarge_as(src : Tensor, other : Tensor) -> Tensor:
    '''
        Add sufficient number of singleton dimensions
        to tensor a **to the right** so to match the
        shape of tensor b. NOTE that simple broadcasting
        works in the opposite direction.
    '''
    return rearrange(src, f'... -> ...{" 1" * (other.dim() - src.dim())}').contiguous()

def pick_frames(
    video : Tensor,
    frames_idxs : Tensor | None = None,
    frames_per_batch : int | None = None,
) -> Tensor:
    '''
        Randomly pick a subset of frames from the input video
        tensor. The number of frames to pick is determined by
        the `frames_per_batch` parameter.
    '''
    assert exists(frames_idxs) ^ exists(frames_per_batch), 'Either `frames_idxs` or `frames_per_batch` must be provided.'
    
    b, c, t, h, w = video.shape
    
    # Randomly sample the indices of the frames to pick
    frame_idxs = default(frames_idxs, torch.cat([
            torch.randperm(t, device=video.device)[:frames_per_batch]
            for _ in range(b)]
        )
    )
    
    batch_idxs = torch.repeat_interleave(
        torch.arange(b, device=video.device),
        default(frames_per_batch, frame_idxs.numel() // b)
    )
    
    return video[batch_idxs, :, frame_idxs, ...]

def enc2dec_name(name : str) -> str:
    return name.replace('downsample', 'upsample')
    
def default_iterdata_worker_init(worker_id : int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()
    
    if worker_info is None: return
    
    dataset = worker_info.dataset
    glob_start = dataset._start # type: ignore
    glob_end   = dataset._end   # type: ignore
    
    per_worker = int((glob_end - glob_start) / worker_info.num_workers)
    worker_id = worker_info.id
    
    dataset._start = glob_start + worker_id * per_worker        # type: ignore 
    dataset._end   = min(dataset._start + per_worker, glob_end) # type: ignore
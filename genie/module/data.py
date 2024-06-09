import yaml
from abc import abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from lightning import LightningDataModule

from typing import Callable

from genie.utils import default
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
            
        return DataLoader(
            self.test__dataset,                  # type: ignore
            sampler        = self.test__sampler, # type: ignore
            batch_size     = self.val_batch_size,
            shuffle        = self.val_shuffle,
            collate_fn     = self.collate_fn,
            num_workers    = self.num_workers,
            worker_init_fn = worker_init_fn,
        )

import torch.nn as nn

from lightning import LightningModule

class VideoTokenizer(LightningModule):
    '''
    Video Tokenizer based on the MagViT-2 paper:
    "Language Model Beats Diffusion: Tokenizer is
    key to visual generation", Yu et al. (2024).
    '''
    
    def __init__(
        self, 
    
    ) -> None:
        super().__init__()
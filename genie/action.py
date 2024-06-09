import torch.nn as nn

from genie.module.quantization import LookupFreeQuantization

class LatentAction(nn.Module):
    '''Latent Action Model (LAM) used to distill latent actions
    from history of past video frames. The LAM model employs a
    VQ-VAE model to encode video frames into discrete latents.
    Both the encoder and decoder are based on spatial-temporal
    transformers.
    '''
    
    def __init__(
        self,
        num_layers: int,
        d_codebook: int,
    ) -> None:
        super().__init__()
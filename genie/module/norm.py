import torch.nn as nn

class AdaptiveGroupNorm(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
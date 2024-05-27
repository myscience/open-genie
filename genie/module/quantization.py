import torch
import torch.nn as nn
from torch import log

from torch import Tensor
from einops import reduce
from einops import einsum
from einops import rearrange
from einops import pack, unpack

from torch.nn.functional import mse_loss

from typing import Tuple

from genie.utils import default

def entropy(p : Tensor, eps : float = 1e-6) -> Tensor:
    '''Calculates the entropy of a probability distribution.

    Args:
        p (Tensor): The probability distribution.
        eps (float, optional): A small value to avoid taking the logarithm of zero.
            Defaults to 1e-6.

    Returns:
        Tensor: The entropy of the probability distribution.
    '''
    return - (p * log(p.clamp(min=eps))).sum(dim=-1)

# Simplified version of the lucidrains implementation at:
# https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/lookup_free_quantization.py#L49
class LookupFreeQuantization(nn.Module):
    '''
    Lookup-Free Quantization module as originally introduced
    in the paper "Language Model Beats Diffusion: Tokenizer
    is key to visual generation" Yu et al. (2024).
    '''
    
    def __init__(
        self,
        d_codebook : int,
        n_codebook : int = 1,
        input_dim : int | None = None,
        use_bias : bool = True,
        frac_sample : float = 1.,
        commit_weight : float = 0.25,
        entropy_weight : float = 0.1,
        diversity_weight : float = 1.,
    ) -> None:
        super().__init__()
        
        input_dim = default(input_dim, d_codebook * n_codebook)
        
        project = input_dim != d_codebook * n_codebook
        
        self.proj_inp = nn.Linear(input_dim, d_codebook * n_codebook, bias=use_bias) if project else nn.Identity()
        self.proj_out = nn.Linear(d_codebook * n_codebook, input_dim, bias=use_bias) if project else nn.Identity()
        
        self.n_codebook = n_codebook
        self.frac_sample = frac_sample
        self.commit_weight = commit_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        
        # * Initialize the codebook
        # Use the bit_mask to generate the bit-codes for all the codebook entries
        # and then convert them to the actual codebook values {-1, 1}. Resulting
        # codebook will have shape (2 ** d_codebook, d_codebook).
        self.register_buffer('bit_mask', 2 ** torch.arange(d_codebook - 1, -1, -1))
        
        codes = torch.arange(2 ** d_codebook, dtype=int)[:, None] & self.bit_mask
        self.register_buffer('codebook', 2 * (codes != 0).float() - 1, persistent=False)
        
    def forward(
        self,
        inp : Tensor,
        beta : float = 100.,
        transpose : bool = False
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor | None]:
        
        # Standardize the input tensor to have shape (batch_size, seq_len, inp_dim)
        inp = rearrange(inp, 'b d ... -> b ... d') if transpose else inp
        inp, ps = pack([inp], 'b * d')
        
        inp = self.proj_inp(inp)
        
        # Split into n_codebook parts
        inp = rearrange(inp, 'b n (c d) -> b n c d', c=self.n_codebook)
        
        # Quantize by simply assigning {-1, 1} to the input tensor depending on the sign
        # of the input tensor values. This is the lookup-free quantization step.
        # See Eq. (3) in the original paper. To obtain the quantized-code indices
        # we simply sum the bit-codes representation of the quantized values.
        quant = inp.sign()
        idxs = reduce((inp > 0).int() * self.bit_mask.int(), 'b n c d -> b n c', 'sum')
        
        # Use straight-through estimator to back-propagate through the quantization step
        code = (inp + (quant - inp).detach()) if self.training else quant
        code = rearrange(code, 'b n c d -> b n (c d)')
        
        # Reconstruct the input tensor from the quantized values
        out = self.proj_out(code)
        out = unpack(out, ps, 'b * d')[0]
        out = rearrange(out, 'b ... d -> b d ...') if transpose else out
        
        # NOTE: Squeeze to remove the n_codebook dimension
        idxs = unpack(idxs, ps, 'b * d')[0].squeeze()
        idxs = rearrange(idxs, 'b ... d -> b d ...') if transpose else idxs
        
        # No need to compute the loss if we are not training
        if not self.training: return (out, idxs), None
        
        # Compute the entropy loss
        inp_prob = 2 * einsum(inp, self.codebook, '... i d, j d -> ... i j')
        inp_prob = (inp_prob * beta).softmax(dim=-1)
        inp_prob = rearrange(inp_prob, 'b n ... -> (b n) ...')
        
        avg_prob = reduce(inp_prob, '... c d -> c d', 'mean')
        
        inp_ent = entropy(inp_prob).mean()
        avg_ent = entropy(avg_prob).mean()
        
        entropy_loss = inp_ent + self.diversity_weight * avg_ent
        
        # Compute commitment loss
        commit_loss = mse_loss(inp, quant.detach(), reduction = 'mean')
        
        # Compute the complete final loss
        loss = entropy_loss * self.entropy_weight + commit_loss * self.commit_weight
        
        return (out, idxs), loss
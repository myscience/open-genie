from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor, softmax
from torch.nn.functional import cross_entropy

from genie.utils import Blueprint
from genie.module import parse_blueprint

class DynamicsModel(nn.Module):
    '''Dynamics Model (DM) used to predict future video frames
    given the history of past video frames and the latent actions.
    The DM model employs the Mask-GIT architecture as introduced
    in Chang et al. (2022).
    '''
    
    def __init__(
        self,
        desc: Blueprint,
        vocab_size: int,
        d_codebook: int,
    ) -> None:
        super().__init__()
        
        self.dec_layers, self.ext_kw = parse_blueprint(desc)
        
        self.embed = nn.Embedding(d_codebook, vocab_size)
        self.head = nn.Linear(vocab_size, vocab_size)
        
    def forward(
        self,
        prev_tok : Tensor,
        tok_idxs : Tensor,
        act_idxs : Tensor,
        tok_mask : Tensor,
        mask_val : float = 0.,
    ) -> Tensor:
        '''
        Predicts the next video token based on the previous tokens
        '''
        
        # Actions are quantized, use them as additive embeddings to tokens
        # Token have shape (batch, seq_len, token_dim)
        pred_tok = prev_tok + self.embed(act_idxs)
        
        # Mask tokens based on external mask as training signal
        pred_tok = torch.masked_fill(pred_tok, rearrange(tok_mask, 'n -> 1 n 1'), 0.)
        
        # Predict the next video token based on previous tokens and actions
        for dec, has_ext in zip(self.dec_layers, self.ext_kw):
            pred_tok = dec(pred_tok)
            
        # Compute the next token probability
        pred_tok = self.head(pred_tok)
        
        # Rearrange tokens to have shape (batch * seq_len, vocab_size)
        pred_tok = rearrange(pred_tok[:, :-1], 'b n v -> (b s) v')
        next_tok = rearrange(tok_idxs[:, +1:], 'b n   -> (b s)')
        
        # Compute the cross-entropy loss between the predicted and actual tokens
        loss = cross_entropy(pred_tok, next_tok)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        prev_tok : Tensor,
        act_idxs : Tensor,
        temperature : float = 1.,
        use_top_k : int = 50,
    ) -> Tensor:
        '''
        Generates future video tokens based on the previous tokens
        and actions
        '''
        *_, d = prev_tok.shape
        
        # Actions are quantized, use them as additive embeddings to tokens
        # Token have shape (batch, seq_len, token_dim)
        pred_tok = prev_tok + self.embed(act_idxs)
        
        # Predict the next video token based on previous tokens and actions
        for dec, has_ext in zip(self.dec_layers, self.ext_kw):
            pred_tok = dec(pred_tok)
            
        # Compute the next token probability
        logits = self.head(pred_tok)
        
        # Rearrange tokens to have shape (batch * seq_len, vocab_size)
        pred_tok = rearrange(pred_tok[:, :-1], 'b n v -> (b s) v')
        
        # Get the token with the highest probability by zeroing out
        # the probability of the lowest probability tokens
        tok_prob = softmax(logits / temperature, dim=-1)
        idxs = tok_prob.topk(k=d - use_top_k, largest=False, sorted=False).indices
        tok_prob.scatter_(dim=-1, index=idxs, src=torch.zeros_like(tok_prob))
        tok_prob /= tok_prob.sum(dim=-1, keepdim=True)
        
        # Generate the next token based on the predicted token probabilities
        next_tok = torch.multinomial(tok_prob, num_samples=1, replacement=True).squeeze()
        
        return next_tok
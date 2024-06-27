from math import inf, pi
from typing import Literal
from einops import pack, rearrange, unpack
import torch
import torch.nn as nn
from torch import Tensor, softmax
from torch.nn.functional import cross_entropy

from genie.utils import Blueprint, exists
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
        
        self.vocab_size = vocab_size
        
    def forward(
        self,
        prev_tok : Tensor,
        act_idxs : Tensor,
    ) -> Tensor:
        '''
        Predicts the next video token based on the previous tokens
        '''
        
        # Actions are quantized, use them as additive embeddings to tokens
        # Token have shape (batch, seq_len, token_dim)
        pred_tok = prev_tok + self.embed(act_idxs)
        
        # Predict the next video token based on previous tokens and actions
        for dec, has_ext in zip(self.dec_layers, self.ext_kw):
            pred_tok = dec(pred_tok)
            
        # Compute the next token probability
        logits = self.head(pred_tok)
        
        return logits
    
    def compute_loss(
        self,
        tokens : Tensor,
        tok_id : Tensor,
        act_id : Tensor,
        mask : Tensor | None = None,
        fill : float = 0.,
    ) -> Tensor:
        
        # Mask tokens based on external mask as training signal
        if exists(mask):
            pred_tok = torch.masked_fill(pred_tok, rearrange(mask, 'n -> 1 n 1'), fill)
        
        # Compute the model prediction for the next token
        logits = self(tokens, act_id)
        
        # Rearrange tokens to have shape (batch * seq_len, vocab_size)
        logits = rearrange(logits[:, :-1], 'b n v -> (b s) v')
        target = rearrange(tok_id[:, +1:], 'b n   -> (b s)')
        
        # Compute the cross-entropy loss between the predicted and actual tokens
        loss = cross_entropy(logits, target)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        prev_tok : Tensor,
        act_idxs : Tensor,
        steps : int = 10,
        which : Literal['linear', 'cosine', 'arccos'] = 'linear',
        temp : float = 1.,
        topk : int = 50,
        masked_tok : float = 0,
    ) -> Tensor:
        '''
        Given past token and action history, predicts the next token
        via the Mask-GIT sampling technique.
        '''
        b, d, t, h, w = prev_tok.shape
        
        # Get the sampling schedule
        schedule = self.get_schedule(steps, which=which)
        
        # Initialize a fully active mask to signal that all the tokens
        # must receive a prediction. The mask will be updated at each
        # step based on the sampling schedule.
        mask = torch.ones(b, h, w, dtype=bool, device=prev_tok.device)
        code = torch.full((b, d, h, w), masked_tok, device=prev_tok.device)
        
        tokens, ps = pack([prev_tok, code], 'b d * h w')
        
        for num_tokens in schedule:
            # If no more tokens to predict, return
            if mask.sum() == 0: break
            
            # Get prediction for the next tokens
            logits = self(tokens, act_idxs)
            
            # Get the logits for the last time-step token
            _, logits = unpack(logits, ps, 'b d * h w -> b d * h w')
            
            logits = rearrange(logits, 'b d h w -> b h w d')
            # Refine the mask based on the sampling schedule
            prob = softmax(logits / temp, dim=-1)
            idxs = torch.multinomial(prob, num_samples=1, replacement=True).squeeze()
            conf = torch.gather(prob, -1, idxs)
            
            # We paint the t-tokens with highest confidence, excluding the
            # already predicted tokens from the mask
            conf[~mask.bool()] = -inf
            vals, idxs = torch.topk(conf.view(b, -1), k=t, dim=-1)
            
            vals = rearrange(vals, 'b (h w) -> b h w', h=h, w=w)
            idxs = rearrange(idxs, 'b (h w) -> b h w', h=h, w=w)
            
            # Fill the code with sampled tokens & update mask
            code.scatter_(1, idxs, vals)
            mask[idxs] = False
            
            tokens, ps = pack([prev_tok, code], 'b d * h w')
        
        return tokens
    
    def get_schedule(
        self,
        steps: int,
        which: Literal['linear', 'cosine', 'arccos'] = 'linear',
    ) -> Tensor:
        t = torch.linspace(1, 0, steps)
        
        match which:
            case 'linear':
                s = 1 - t
            case 'cosine':
                s = torch.cos(t * pi * .5)
            case 'arccos':
                s = torch.acos(t) / (pi * .5)
            case _:
                raise ValueError(f'Unknown schedule type: {which}')
        
        # Fill the schedule with the ratio of tokens to predict
        schedule = (s / s.sum()) * self.vocab_size
        schedule = schedule.round().int().clamp(min=1)
        
        # Make sure that the total number of tokens to predict is
        # equal to the vocab size
        schedule[-1] += self.vocab_size - schedule.sum()
        
        return schedule
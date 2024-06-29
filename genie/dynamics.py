import torch
import torch.nn as nn
from typing import Literal
from math import inf, pi, prod
from torch import Tensor, softmax
from torch.nn.functional import cross_entropy

from einops import pack, rearrange, unpack
from einops.layers.torch import Rearrange

from genie.utils import Blueprint, default
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
        tok_vocab: int,
        act_vocab: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        
        self.dec_layers, self.ext_kw = parse_blueprint(desc)
        
        self.head = nn.Linear(embed_dim, tok_vocab)
        
        self.tok_emb = nn.Embedding(tok_vocab, embed_dim)
        self.act_emb = nn.Sequential(
            nn.Embedding(act_vocab, embed_dim),
            Rearrange('b t d -> b t 1 1 d'),
        )
        
        self.tok_vocab = tok_vocab
        self.act_vocab = act_vocab
        self.embed_dim = embed_dim
        
    def forward(
        self,
        tokens : Tensor,
        act_id : Tensor,
    ) -> Tensor:
        '''
        Predicts the next video token based on the previous tokens
        '''
        
        # Actions are quantized, use them as additive embeddings to tokens
        # Token have shape (batch, seq_len, token_dim)
        tokens = self.tok_emb(tokens) + self.act_emb(act_id)
        
        # Predict the next video token based on previous tokens and actions
        for dec, has_ext in zip(self.dec_layers, self.ext_kw):
            tokens = dec(tokens)
        
        # Compute the next token probability
        logits = self.head(tokens)
        
        return logits, logits[:, -1]
    
    def compute_loss(
        self,
        tokens : Tensor,
        act_id : Tensor,
        mask : Tensor | None = None,
        fill : float = 0.,
    ) -> Tensor:
        
        b, t, h, w = tokens.shape
        
        # Create Bernoulli mask if not provided
        mask = default(mask, torch.distributions.Bernoulli(
            torch.empty(1).uniform_(0.5, 1).item() # Random rate in [0.5, 1]
            ).sample((b, t, h, w)).bool()
        )
        
        # Mask tokens based on external mask as training signal
        tokens = torch.masked_fill(tokens, mask, fill)
        
        # Compute the model prediction for the next token
        logits, _ = self(tokens, act_id.detach())
        
        # Only compute loss on the tokens that were masked
        logits = logits[mask.squeeze()]
        tokens = tokens[mask.squeeze()]
        
        # Rearrange tokens to have shape (batch * seq_len, vocab_size)
        logits = rearrange(logits, '... d -> (...) d')
        target = rearrange(tokens, '...   -> (...)')
        
        # Compute the cross-entropy loss between the predicted and actual tokens
        loss = cross_entropy(logits, target)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        tokens : Tensor,
        act_id : Tensor,
        steps : int = 10,
        which : Literal['linear', 'cosine', 'arccos'] = 'linear',
        temp : float = 1.,
        topk : int = 50,
        masked_tok : int = 0,
    ) -> Tensor:
        '''
        Given past token and action history, predicts the next token
        via the Mask-GIT sampling technique.
        '''
        b, t, h, w = tokens.shape
        
        # Get the sampling schedule
        schedule = self.get_schedule(steps, shape=(h, w), which=which)
        
        # Initialize a fully active mask to signal that all the tokens
        # must receive a prediction. The mask will be updated at each
        # step based on the sampling schedule.
        mask = torch.ones(b, h, w, dtype=bool, device=tokens.device)
        code = torch.full((b, h, w), masked_tok, device=tokens.device)
        mock = torch.zeros(b, dtype=int, device=tokens.device)
        
        tok_id, ps = pack([tokens, code], 'b * h w')
        act_id, _ = pack([act_id, mock], 'b *')
        
        for num_tokens in schedule:
            # If no more tokens to predict, return
            if mask.sum() == 0: break
            
            # Get prediction for the next tokens
            _, logits = self(tok_id, act_id)
            
            # Refine the mask based on the sampling schedule
            prob = softmax(logits / temp, dim=-1)
            prob, ps = pack([prob], '* d')
            pred = torch.multinomial(prob, num_samples=1)
            conf = torch.gather(prob, -1, pred)
            conf = unpack(conf, ps, '* d')[0].squeeze()
            
            # We paint the k-tokens with highest confidence, excluding the
            # already predicted tokens from the mask
            conf[~mask.bool()] = -inf
            idxs = torch.topk(conf.view(b, -1), k=num_tokens, dim=-1).indices
            
            code, cps = pack([code], 'b *')
            mask, mps = pack([mask], 'b *')
            pred = pred.view(b, -1)
            
            # Fill the code with sampled tokens & update mask
            vals = torch.gather(pred, -1, idxs).to(code.dtype)
            code.scatter_(1, idxs, vals)
            mask.scatter_(1, idxs, False)
            
            code = unpack(code, cps, 'b *')[0]
            mask = unpack(mask, mps, 'b *')[0]
            
            pred_tok, ps = pack([tokens, code], 'b * h w')
            
        assert mask.sum() == 0, f'Not all tokens were predicted. {mask.sum()} tokens left.'
        return pred_tok
    
    def get_schedule(
        self,
        steps: int,
        shape: tuple[int, int],
        which: Literal['linear', 'cosine', 'arccos'] = 'linear',
    ) -> Tensor:
        n = prod(shape)
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
        schedule = (s / s.sum()) * n
        schedule = schedule.round().int().clamp(min=1)
        
        # Make sure that the total number of tokens to predict is
        # equal to the vocab size
        schedule[-1] += n - schedule.sum()
        
        return schedule
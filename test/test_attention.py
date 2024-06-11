import unittest

import torch

from genie.module.attention import Attention
from genie.module.attention import SpatialAttention
from genie.module.attention import TemporalAttention
from genie.module.attention import SpaceTimeAttention

class TestAttentionModule(unittest.TestCase):
    def setUp(self) -> None:
        self.n_embd = 16
        self.n_head = 4
        self.d_head = 32
        self.bias = True
        self.cond_dim = 8
    
    def test_self_attention(self):
        # Test SelfAttention class
        attn = Attention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
        )
        
        mock_seq = torch.randn(2, 16, self.n_embd)
        
        attn_out = attn(mock_seq)
        
        self.assertEqual(attn_out.shape, mock_seq.shape)
        
    def test_cross_attention(self):
        # Test SelfAttention class
        attn = Attention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
            key_dim = self.cond_dim,
        )
        
        mock_seq  = torch.randn(2, 16, self.n_embd)
        mock_cond = torch.randn(2, 16, self.cond_dim)
        
        attn_out = attn(mock_seq, mock_cond)
        
        self.assertEqual(attn_out.shape, mock_seq.shape)
        
    def test_spatial_attention_image(self):
        # Test SpatialAttention class
        attn = SpatialAttention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
        )
        
        mock_img = torch.randn(2, self.n_embd, 32, 32)
        
        attn_out = attn(mock_img)
        
        self.assertEqual(attn_out.shape, mock_img.shape)
        
    def test_spatial_attention_video(self):
        # Test SpatialAttention class
        attn = SpatialAttention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
        )
        
        mock_img = torch.randn(2, self.n_embd, 16, 32, 32)
        
        attn_out = attn(mock_img)
        
        self.assertEqual(attn_out.shape, mock_img.shape)
        
    def test_temporal_attention_video(self):
        # Test SpatialAttention class
        attn = TemporalAttention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
        )
        
        mock_img = torch.randn(2, self.n_embd, 16, 32, 32)
        
        attn_out = attn(mock_img)
        
        self.assertEqual(attn_out.shape, mock_img.shape)
        
class TestSpaceTimeAttention(unittest.TestCase):
    def setUp(self):
        self.n_embed = 256
        
        self.action_block = SpaceTimeAttention(
            n_embd=self.n_embed,
            n_head=(4, 4),
            d_head=(32, 32),
            hid_dim=(512, 512),
            bias=True,
            embed=True,
            scale=0.5,
            dropout=0.1,
        )

    def test_forward(self):
        inp_video = torch.randn(1, self.n_embed, 16, 32, 32)
        out_video = self.action_block(inp_video)
        
        self.assertEqual(out_video.shape, (1, self.n_embed, 16, 32, 32))

if __name__ == '__main__':
    unittest.main()
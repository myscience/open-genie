import unittest

import torch

from genie.module.attention import SelfAttention
from genie.module.attention import SpatialAttention
from genie.module.attention import TemporalAttention

class TestAttentionModule(unittest.TestCase):
    def setUp(self) -> None:
        self.n_embd = 16
        self.n_head = 4
        self.d_head = 32
        self.bias = True
    
    def test_self_attention(self):
        # Test SelfAttention class
        attn = SelfAttention(
            n_embd = self.n_embd,
            n_head = self.n_head,
            d_head = self.d_head,
            bias   = self.bias,
            causal = True,
        )
        
        mock_seq = torch.randn(2, 16, self.n_embd)
        
        attn_out = attn(mock_seq)
        
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

if __name__ == '__main__':
    unittest.main()
import unittest

import torch
from genie.module.quantization import LookupFreeQuantization

class TestLookupFreeQuantization(unittest.TestCase):
    def setUp(self) -> None:
        self.d_codebook       =   8 # Codebook quantization, i.e. codebook size 2 ** d_codebook
        self.input_dim        = 256 # Expected input dimension
        self.use_bias         = True
        self.frac_sample      = .8
        self.commit_weight    = .25
        self.entropy_weight   = .1
        self.diversity_weight = 1.
        self.batch_size       = 4
        
        # Create mock input tensor
        self.seq_len = 16
        self.input_tensor = torch.rand((self.batch_size, self.seq_len, self.input_dim))
    
    def test_eval_quantize_single_codebook(self):
        num_codebooks = 1
        lfq = LookupFreeQuantization(
            codebook_dim     = self.d_codebook,
            num_codebook     = num_codebooks,
            input_dim        = self.input_dim,
            use_bias         = self.use_bias,
            frac_sample      = self.frac_sample,
            commit_weight    = self.commit_weight,
            entropy_weight   = self.entropy_weight,
            diversity_weight = self.diversity_weight
        )
        
        self.assertEqual(lfq.num_codebooks, 1)
        self.assertEqual(lfq.frac_sample, self.frac_sample)
        self.assertEqual(lfq.commit_weight, self.commit_weight)
        self.assertEqual(lfq.entropy_weight, self.entropy_weight)
        self.assertEqual(lfq.diversity_weight, self.diversity_weight)
        
        # Quantize the input tensor
        lfq.eval() # Only the test quantization
        (quant, idxs), _ = lfq(self.input_tensor)
        
        # Check the shape of the quantized tensor
        self.assertEqual(quant.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual( idxs.shape, (self.batch_size, self.seq_len)) # NOTE: No num_codebooks dimension
        
        if self.input_dim == self.d_codebook:
            # If not output projection, check that tokens have values in {-1, +1}
            self.assertTrue(torch.allclose(quant, torch.sign(quant)))
        
    def test_train_quantize_single_codebook(self):
        num_codebooks = 1
        lfq = LookupFreeQuantization(
            codebook_dim       = self.d_codebook,
            num_codebook       = num_codebooks,
            input_dim        = self.input_dim,
            use_bias         = self.use_bias,
            frac_sample      = self.frac_sample,
            commit_weight    = self.commit_weight,
            entropy_weight   = self.entropy_weight,
            diversity_weight = self.diversity_weight
        )
        
        self.assertEqual(lfq.num_codebooks, 1)
        self.assertEqual(lfq.frac_sample, self.frac_sample)
        self.assertEqual(lfq.commit_weight, self.commit_weight)
        self.assertEqual(lfq.entropy_weight, self.entropy_weight)
        self.assertEqual(lfq.diversity_weight, self.diversity_weight)
        
        # Quantize the input tensor
        lfq.train() # Only the test quantization
        (quant, idxs), loss = lfq(self.input_tensor)
        
        # Check the shape of the quantized tensor
        self.assertEqual(quant.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual( idxs.shape, (self.batch_size, self.seq_len)) # NOTE: No num_codebooks dimension
        
        self.assertGreater(loss, 0.)
        
    def test_train_quantize_multi_codebook(self):
        num_codebooks = 3
        lfq = LookupFreeQuantization(
            codebook_dim       = self.d_codebook,
            num_codebook       = num_codebooks,
            input_dim        = self.input_dim,
            use_bias         = self.use_bias,
            frac_sample      = self.frac_sample,
            commit_weight    = self.commit_weight,
            entropy_weight   = self.entropy_weight,
            diversity_weight = self.diversity_weight
        )
        
        self.assertEqual(lfq.num_codebooks, num_codebooks)
        self.assertEqual(lfq.frac_sample, self.frac_sample)
        self.assertEqual(lfq.commit_weight, self.commit_weight)
        self.assertEqual(lfq.entropy_weight, self.entropy_weight)
        self.assertEqual(lfq.diversity_weight, self.diversity_weight)
        
        # Quantize the input tensor
        lfq.train() # Only the test quantization
        (quant, idxs), loss = lfq(self.input_tensor)
        
        # Check the shape of the quantized tensor
        self.assertEqual(quant.shape, (self.batch_size, self.seq_len, self.input_dim))
        self.assertEqual( idxs.shape, (self.batch_size, self.seq_len, num_codebooks))
        
        self.assertGreater(loss, 0.)

if __name__ == '__main__':
    unittest.main()
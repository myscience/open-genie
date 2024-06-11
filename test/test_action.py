import unittest
import torch
from genie.action import LatentAction

class TestLatentAction(unittest.TestCase):
    def setUp(self):
        self.latent_action = LatentAction(
            num_layers=4,
            d_codebook=8,
            n_embd=256,
            n_head=(4, 4),
            d_head=(32, 32),
            ff_hid_dim=(512, 512),
            dropout=0.1,
            n_codebook=1,
            lfq_bias=True,
            lfq_frac_sample=1.0,
            lfq_commit_weight=0.25,
            lfq_entropy_weight=0.1,
            lfq_diversity_weight=1.0
        )

    def test_forward(self):
        video = torch.randn(1, 3, 16, 32, 32)
        recon, q_loss = self.latent_action(video)
        self.assertEqual(recon.shape, (1, 3, 16, 32, 32))
        self.assertEqual(q_loss.shape, (1,))

if __name__ == '__main__':
    unittest.main()
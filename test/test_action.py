import unittest
import torch
from genie.action import LatentAction

ACTION_BLUEPRINT = (
    ('space-time_attn', {
        'n_rep' : 6,
        'n_embd' : 256,
        'n_head' : 8,
        'd_head' : 16,
    }),
)

class TestLatentAction(unittest.TestCase):
    def setUp(self):
        self.blueprint = ACTION_BLUEPRINT
        self.d_codebook = 8
        self.inp_channels = 3
        self.ker_size = 3
        self.n_embd = 256
        self.n_codebook = 1
        self.lfq_bias = True
        self.lfq_frac_sample = 1.0
        self.lfq_commit_weight = 0.25
        self.lfq_entropy_weight = 0.1
        self.lfq_diversity_weight = 1.0
        
        self.batch_size = 2
        
    def test_encode(self):
        model = LatentAction(
            blueprint=self.blueprint,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            ker_size=self.ker_size,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            lfq_bias=self.lfq_bias,
            lfq_frac_sample=self.lfq_frac_sample,
            lfq_commit_weight=self.lfq_commit_weight,
            lfq_entropy_weight=self.lfq_entropy_weight,
            lfq_diversity_weight=self.lfq_diversity_weight,
        )

        video = torch.randn(self.batch_size, self.inp_channels, 16, 64, 64)
        act, q_loss = model.encode(video)
        
        self.assertEqual(act.shape, (self.batch_size, 16, self.d_codebook))
        self.assertEqual(q_loss.shape, ())
        self.assertTrue(q_loss >= 0)
        
    def test_decode(self):
        model = LatentAction(
            blueprint=self.blueprint,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            ker_size=self.ker_size,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            lfq_bias=self.lfq_bias,
            lfq_frac_sample=self.lfq_frac_sample,
            lfq_commit_weight=self.lfq_commit_weight,
            lfq_entropy_weight=self.lfq_entropy_weight,
            lfq_diversity_weight=self.lfq_diversity_weight,
        )

        video = torch.randn(self.batch_size, self.inp_channels, 16, 64, 64)
        q_act = torch.randint(0, self.d_codebook, (self.batch_size, 16))
        recon, last = model.decode(video, q_act=q_act)
        
        self.assertEqual(recon.shape, (self.batch_size, self.inp_channels, 64, 64))
        self.assertEqual(last.shape,  (self.batch_size, self.inp_channels, 64, 64))

    def test_forward(self):
        model = LatentAction(
            blueprint=self.blueprint,
            d_codebook=self.d_codebook,
            inp_channels=self.inp_channels,
            ker_size=self.ker_size,
            n_embd=self.n_embd,
            n_codebook=self.n_codebook,
            lfq_bias=self.lfq_bias,
            lfq_frac_sample=self.lfq_frac_sample,
            lfq_commit_weight=self.lfq_commit_weight,
            lfq_entropy_weight=self.lfq_entropy_weight,
            lfq_diversity_weight=self.lfq_diversity_weight,
        )

        video = torch.randn(1, self.inp_channels, 16, 64, 64)
        loss, aux_losses = model(video)

        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)

if __name__ == '__main__':
    unittest.main()
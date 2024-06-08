import unittest
import unittest
import torch

from genie import VideoTokenizer

TEST_ENC_DESC = (
    ('causal', {
        'in_channels': 3,
        'out_channels': 64,
        'kernel_size': 3,
    }),
    ('residual', {
        'in_channels': 64,
        'kernel_size': 3,
        'downsample': (1, 2),
        'use_causal': True,
        'use_blur': True,
    }),
    ('residual', {
        'in_channels': 64,
        'out_channels': 128,
    }),
    ('residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('residual', {
        'in_channels': 128,
        'out_channels': 256,
        'kernel_size': 3,
        'downsample': 2,
        'use_causal': True,
    }),
    ('proj_out', {
        'in_channels': 256,
        'out_channels': 18,
        'num_groups': 8,
        'kernel_size': 3,
    })
)

TEST_DEC_DESC = (
    ('causal', {
        'in_channels': 18,
        'out_channels': 128,
        'kernel_size': 3,
    }),
    ('residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('adaptive_group_norm', {
        'num_groups': 8,
        'num_channels': 128,
        'has_ext' : True,
        'dim_cond' : 18,
    }),
    ('residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('spacetime_upsample', {
        'in_channels': 128,
        'kernel_size': 3,
        'time_factor': 2,
        'space_factor': 2,
    }),
    ('adaptive_group_norm', {
        'num_groups': 8,
        'num_channels': 128,
        'has_ext' : True,
        'dim_cond' : 18,
    }),
    ('residual', {
        'in_channels': 128,
        'out_channels': 64,
    }),
    ('residual', {
        'n_rep': 2,
        'in_channels': 64,
    }),
    ('spacetime_upsample', {
        'in_channels': 64,
        'kernel_size': 3,
        'time_factor': 1,
        'space_factor': 2,
    }),
    ('adaptive_group_norm', {
        'num_groups': 8,
        'num_channels': 64,
        'has_ext' : True,
        'dim_cond' : 18,
    }),
    ('proj_out', {
        'in_channels': 64,
        'out_channels': 3,
        'num_groups': 8,
        'kernel_size': 3,
    })
)

class TestVideoTokenizer(unittest.TestCase):
    def setUp(self):
        
        self.d_codebook = 18
        self.n_codebook = 1
        
        self.batch_size = 4
        self.num_frames = 16
        self.num_channels = 3
        self.img_h, self.img_w = 64, 64
        
        self.tokenizer = VideoTokenizer(
            enc_desc = TEST_ENC_DESC,
            dec_desc = TEST_DEC_DESC,
            
            disc_kwargs=dict(
                # Discriminator parameters
                inp_size = (self.img_h, self.img_w),
                model_dim = 64,
                dim_mults = (1, 2, 4),
                down_step = (None, 2, 2),
                inp_channels = self.num_channels,
                kernel_size = 3,
                num_groups = 8,
                num_heads = 4,
                dim_head = 32,
            ),
            
            d_codebook = self.d_codebook,
            n_codebook = self.n_codebook,
            #
            lfq_bias = True,
            lfq_frac_sample = 1.,
            lfq_commit_weight = 0.25,
            lfq_entropy_weight = 0.1,
            lfq_diversity_weight = 1.,
            #
            perceptual_model = 'vgg16',
            perc_feat_layers = ('features.6', 'features.13', 'features.18', 'features.25'),
            gan_discriminate='frames',
            gan_frames_per_batch = 4,
            gan_loss_weight = 1.,
            perc_loss_weight = 1.,
            quant_loss_weight = 1.,
        )
        
        # Example video tensor
        self.video = torch.randn(
            self.batch_size,
            self.num_channels,
            self.num_frames,
            self.img_h,
            self.img_w
        )
        
        self.time_down = 2
        self.space_down = 4

    def test_encode(self):
        encoded = self.tokenizer.encode(self.video)
        self.assertEqual(encoded.shape, (
            self.batch_size,
            self.d_codebook,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down
        ))  # Check output shape

    def test_decode(self):
        quantized = torch.randn(
            self.batch_size,
            self.d_codebook,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down,
        )  # Example quantized tensor
        decoded = self.tokenizer.decode(quantized)
        self.assertEqual(decoded.shape, (
            self.batch_size,
            self.num_channels,
            self.num_frames,
            self.img_h,
            self.img_w
        ))  # Check output shape
        
    def test_tokenize(self):
        tokens, _ = self.tokenizer.tokenize(self.video)
        self.assertEqual(tokens.shape, (
            self.batch_size, 
            self.d_codebook,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down,
        )) # Check output shape

    def test_forward(self):
        loss, aux_losses = self.tokenizer(self.video)
        
        self.assertTrue(loss >= 0)
        for loss in aux_losses:
            self.assertEqual(loss.shape, torch.Size([]))  # Check the output shape
        
        self.assertTrue(aux_losses[0] >= 0)
        self.assertTrue(aux_losses[2] >= 0)
        self.assertTrue(aux_losses[3] >= 0)
        self.assertTrue(aux_losses[4] >= 0)

if __name__ == '__main__':
    unittest.main()
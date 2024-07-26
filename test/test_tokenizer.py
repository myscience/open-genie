from os import path
import unittest

import torch
import yaml

from genie import VideoTokenizer
from genie.dataset import LightningPlatformer2D
from genie.tokenizer import REPR_TOK_ENC
from genie.tokenizer import REPR_TOK_DEC

TEST_ENC_DESC = (
    ('causal-conv3d', {
        'in_channels': 3,
        'out_channels': 64,
        'kernel_size': 3,
    }),
    ('video-residual', {
        'in_channels': 64,
        'kernel_size': 3,
        'downsample': (1, 2),
        'use_causal': True,
        'use_blur': True,
    }),
    ('video-residual', {
        'in_channels': 64,
        'out_channels': 128,
    }),
    ('video-residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('video-residual', {
        'in_channels': 128,
        'out_channels': 256,
        'kernel_size': 3,
        'downsample': 2,
        'use_causal': True,
    }),
    ('causal-conv3d', {
        'in_channels': 256,
        'out_channels': 18,
        'kernel_size': 3,
    })
)

TEST_DEC_DESC = (
    ('causal-conv3d', {
        'in_channels': 18,
        'out_channels': 128,
        'kernel_size': 3,
    }),
    ('video-residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('adaptive_group_norm', {
        'num_groups': 8,
        'num_channels': 128,
        'has_ext' : True,
        'dim_cond' : 18,
    }),
    ('video-residual', {
        'n_rep': 2,
        'in_channels': 128,
    }),
    ('depth2spacetime_upsample', {
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
    ('video-residual', {
        'in_channels': 128,
        'out_channels': 64,
    }),
    ('video-residual', {
        'n_rep': 2,
        'in_channels': 64,
    }),
    ('depth2spacetime_upsample', {
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
    ('causal-conv3d', {
        'in_channels': 64,
        'out_channels': 3,
        'kernel_size': 3,
    })
)

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

with open(local_settings, 'r') as f:
    local_settings = yaml.safe_load(f)

class TestVideoTokenizer(unittest.TestCase):
    def setUp(self):
        
        self.d_codebook = 18
        self.n_codebook = 1
        
        self.batch_size = 2
        self.num_frames = 8
        self.num_channels = 3
        self.img_h, self.img_w = 64, 64
        
        # Number of channels after the encoding by the REPR_TOK_ENC
        self.hid_channels = 512
        
        self.time_down  = 1 # This parameters are determined by REPR_TOK_ENC
        self.space_down = 4 # This parameters are determined by REPR_TOK_ENC
        
        factor = 4
        
        self.tokenizer = VideoTokenizer(
            enc_desc = REPR_TOK_ENC,
            dec_desc = REPR_TOK_DEC,
            
            disc_kwargs=dict(
                # Discriminator parameters
                inp_size = (self.img_h, self.img_w),
                model_dim = 64,
                dim_mults = (1, 2, 4),
                down_step = (None, 2, 2),
                inp_channels = self.num_channels,
                kernel_size = 3,
                use_attn = False,
                use_blur = True,
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

    def test_encode(self):
        encoded = self.tokenizer.encode(self.video)
        self.assertEqual(encoded.shape, (
            self.batch_size,
            self.hid_channels,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down
        ))  # Check output shape

    def test_decode(self):
        quantized = torch.randn(
            self.batch_size,
            self.hid_channels,
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
        tokens, idxs = self.tokenizer.tokenize(self.video)
        self.assertEqual(tokens.shape, (
            self.batch_size, 
            self.hid_channels,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down,
        )) # Check output shape
        
        if self.hid_channels == 2 ** self.d_codebook:
            # If not output projection, check that tokens have values in {-1, +1}
            self.assertTrue(torch.allclose(tokens, torch.sign(tokens)))
        
        self.assertEqual(idxs.shape, (
            self.batch_size,
            self.num_frames // self.time_down,
            self.img_h // self.space_down,
            self.img_w // self.space_down,
        ))

    def test_forward(self):
        loss, aux_losses = self.tokenizer(self.video)
        
        self.assertTrue(loss >= 0)
        for loss in aux_losses:
            self.assertEqual(loss.shape, torch.Size([]))  # Check the output shape
        
        print(aux_losses)
        self.assertTrue(aux_losses[0] >= 0)
        self.assertTrue(aux_losses[2] >= 0)
        self.assertTrue(aux_losses[3] >= 0)
        self.assertTrue(aux_losses[4] >= 0)
        
    def test_forward_platformer_2d(self):
        dataset = LightningPlatformer2D(
            root=local_settings['platformer_remote_root'],
            output_format='c t h w',
            transform=None,
            randomize=True,
            batch_size=self.batch_size,
            num_frames=self.num_frames,
            num_workers=4,
        )

        dataset.setup('fit')
        loader = dataset.train_dataloader()
        
        video = next(iter(loader))

        loss, aux_losses = self.tokenizer(video)
        
        self.assertTrue(loss >= 0)
        for loss in aux_losses:
            self.assertEqual(loss.shape, torch.Size([]))  # Check the output shape
        
        print(aux_losses)
        self.assertTrue(aux_losses[0] >= 0)
        self.assertTrue(aux_losses[2] >= 0)
        self.assertTrue(aux_losses[3] >= 0)
        self.assertTrue(aux_losses[4] >= 0)

if __name__ == '__main__':
    unittest.main()
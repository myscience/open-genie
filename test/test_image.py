import unittest
import torch
from genie.module.image import ResidualBlock
from genie.module.image import SpaceDownsample
from genie.module.image import FrameDiscriminator

class TestImageModule(unittest.TestCase):
    
    def test_space_downsample(self):
        in_dim = 64
        factor = 2
        batch_size = 4
        downsample = SpaceDownsample(in_dim, factor)
        
        inp = torch.randn(batch_size, in_dim, 32, 32)
        out = downsample(inp)
        
        self.assertEqual(out.shape, (batch_size, in_dim, 16, 16))
    
    def test_residual_block_no_downsample(self):
        inp_channel = 64
        out_channel = 128
        kernel_size = 3
        num_groups = 1
        downsample = None
        batch_size = 4
        residual_block = ResidualBlock(
            inp_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            num_groups=num_groups,
            downsample=downsample,
        )
        
        inp = torch.randn(batch_size, inp_channel, 32, 32)
        out = residual_block(inp)
        
        self.assertEqual(out.shape, (batch_size, out_channel, 32, 32))
        
    def test_residual_block_yes_downsample(self):
        inp_channel = 64
        out_channel = 128
        kernel_size = 3
        num_groups = 1
        downsample = 2
        batch_size = 4
        residual_block = ResidualBlock(
            inp_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            num_groups=num_groups,
            downsample=downsample,
        )
        
        img_h, img_w = 64, 64
        
        inp = torch.randn(batch_size, inp_channel, img_h, img_w)
        out = residual_block(inp)
        
        self.assertEqual(out.shape, (batch_size, out_channel, img_h // downsample, img_w // downsample))
    
    def test_frame_discriminator(self):
        frame_size = (64, 64)
        model_dim = 64
        dim_mults = (1, 2, 4)
        down_step = (None, 2, 2)
        inp_channels = 3
        kernel_size = 3
        num_groups = 1
        num_heads = 4
        dim_head = 32
        batch_size = 4
        
        frame_discriminator = FrameDiscriminator(
            frame_size   = frame_size,
            model_dim    = model_dim,
            dim_mults    = dim_mults,
            down_step    = down_step,
            inp_channels = inp_channels,
            kernel_size  = kernel_size,
            num_groups   = num_groups,
            num_heads    = num_heads,
            dim_head     = dim_head
        )
        
        image = torch.randn(batch_size, inp_channels, frame_size[0], frame_size[1])
        out = frame_discriminator(image)
        
        self.assertEqual(out.shape, (batch_size, ))
    
if __name__ == '__main__':
    unittest.main()
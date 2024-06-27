import unittest
import torch
from genie.module.image import BlurPooling2d
from genie.module.image import ImageResidualBlock
from genie.module.image import SpaceDownsample

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
        residual_block = ImageResidualBlock(
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
        residual_block = ImageResidualBlock(
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
    
    def test_blur_pooling(self):
        kernel_size = 3
        batch_size = 4
        inp_channel = 64
        stride = 2
        img_h, img_w = 32, 32
        
        blur_pooling = BlurPooling2d(
            kernel_size,
            stride=stride,
        )
        
        inp = torch.randn(batch_size, inp_channel, img_h, img_w)
        out = blur_pooling(inp)
        
        self.assertEqual(out.shape, (batch_size, inp_channel, img_h // stride, img_w // stride))
    
if __name__ == '__main__':
    unittest.main()
import unittest
import torch
from genie.module.video import CausalConv3d
from genie.module.video import CausalConvTranspose3d
from genie.module.video import DepthToSpaceUpsample
from genie.module.video import DepthToTimeUpsample
from genie.module.video import SpaceTimeDownsample
from genie.module.video import SpaceTimeUpsample
from genie.module.video import ResidualBlock

class TestVideoModule(unittest.TestCase):
    def test_causal_conv3d(self):
        # Create a CausalConv3d instance
        conv = CausalConv3d(3, 64, kernel_size=3)
        
        # Create a random input tensor
        inp = torch.randn(1, 3, 16, 16, 16)
        
        # Perform forward pass
        out = conv(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 64, 16, 16, 16))
        
    def test_causal_conv_transpose3d(self):
        # Create a CausalConvTranspose3d instance
        conv_transpose = CausalConvTranspose3d(64, 3, kernel_size=3)
        
        # Create a random input tensor
        inp = torch.randn(1, 64, 16, 16, 16)
        
        # Perform forward pass
        out = conv_transpose(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 3, 16, 16, 16))
        
    def test_space_upsample(self):
        # Create a SpaceUpsample instance
        upsample = DepthToSpaceUpsample(64, factor=2)
        
        # Create a random input tensor
        inp = torch.randn(1, 64, 8, 16, 16)
        
        # Perform forward pass
        out = upsample(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 64, 8, 32, 32))
        
    def test_time_upsample(self):
        # Create a TimeUpsample instance
        upsample = DepthToTimeUpsample(64, factor=2)
        
        # Create a random input tensor
        inp = torch.randn(1, 64, 8, 16, 16)
        
        # Perform forward pass
        out = upsample(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 64, 16, 16, 16))
        
    def test_space_time_downsample(self):
        # Create a SpaceTimeDownsample instance
        downsample = SpaceTimeDownsample(64, 128, time_factor=2, space_factor=2)
        
        # Create a random input tensor
        inp = torch.randn(1, 64, 16, 28, 28)
        
        # Perform forward pass
        out = downsample(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 128, 8, 14, 14))
        
    def test_space_time_upsample(self):
        # Create a SpaceTimeUpsample instance
        upsample = SpaceTimeUpsample(128, 64, time_factor=2, space_factor=2)
        
        # Create a random input tensor
        inp = torch.randn(1, 128, 8, 7, 7)
        
        # Perform forward pass
        out = upsample(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 64, 16, 14, 14))
        
    def test_residual_block(self):
        # Create a ResidualBlock instance
        block = ResidualBlock(
            inp_channel=64,
            out_channel=128,
        )
        
        # Create a random input tensor
        inp = torch.randn(1, 64, 8, 16, 16)
        
        # Perform forward pass
        out = block(inp)
        
        # Check output shape
        self.assertEqual(out.shape, (1, 128, 8, 16, 16))
        
if __name__ == '__main__':
    unittest.main()
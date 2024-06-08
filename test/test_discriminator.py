import unittest
import torch
from genie.module.discriminator import FrameDiscriminator, VideoDiscriminator

class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.time_size = 16
        self.frame_size = (64, 64)
        self.model_dim = 64
        self.dim_mults = (1, 2, 4)
        self.down_step = (None, 2, 2)
        self.inp_channels = 3
        self.kernel_size = 3
        self.num_groups = 1
        self.num_heads = 4
        self.dim_head = 32
        
        self.batch_size = 2

    def test_frame_discriminator(self):
        discriminator = FrameDiscriminator(
            frame_size=self.frame_size,
            model_dim=self.model_dim,
            dim_mults=self.dim_mults,
            down_step=self.down_step,
            inp_channels=self.inp_channels,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            num_heads=self.num_heads,
            dim_head=self.dim_head
        )
        input_tensor = torch.randn(self.batch_size, self.inp_channels, self.frame_size[0], self.frame_size[1])
        output_tensor = discriminator(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, ))
        
    def test_frame_discriminator_attn(self):
        discriminator = FrameDiscriminator(
            frame_size=self.frame_size,
            model_dim=self.model_dim,
            dim_mults=self.dim_mults,
            down_step=self.down_step,
            inp_channels=self.inp_channels,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            use_attn=True,
        )
        frame_tensor = torch.randn(self.batch_size, self.inp_channels, self.frame_size[0], self.frame_size[1])
        output_tensor = discriminator(frame_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, ))

    def test_video_discriminator(self):
        discriminator = VideoDiscriminator(
            video_size=(self.time_size, *self.frame_size),
            model_dim=self.model_dim,
            dim_mults=self.dim_mults,
            down_step=self.down_step,
            inp_channels=self.inp_channels,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            num_heads=self.num_heads,
            dim_head=self.dim_head
        )
        
        video_tensor = torch.randn(self.batch_size, self.inp_channels, self.time_size, self.frame_size[0], self.frame_size[1])
        output_tensor = discriminator(video_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, ))
        
    def test_video_discriminator_attn(self):
        discriminator = VideoDiscriminator(
            video_size=(self.time_size, *self.frame_size),
            model_dim=self.model_dim,
            dim_mults=self.dim_mults,
            down_step=self.down_step,
            inp_channels=self.inp_channels,
            kernel_size=self.kernel_size,
            num_groups=self.num_groups,
            num_heads=self.num_heads,
            dim_head=self.dim_head,
            use_attn=True,
        )
        
        video_tensor = torch.randn(self.batch_size, self.inp_channels, self.time_size, self.frame_size[0], self.frame_size[1])
        output_tensor = discriminator(video_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, ))

if __name__ == '__main__':
    unittest.main()
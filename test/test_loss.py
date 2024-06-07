import unittest
import torch
from genie.module.loss import PerceptualLoss, GANLoss

class TestLossModule(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_size = 2
        self.num_channels = 3
        self.num_frames = 4
        self.img_h, self.img_w = 64, 64
        self.inp_video = torch.randn(
            self.batch_size,
            self.num_channels,
            self.num_frames,
            self.img_h,
            self.img_w
        ) # Mock real input video tensor
        
        self.rec_video = torch.randn(
            self.batch_size,
            self.num_channels,
            self.num_frames,
            self.img_h,
            self.img_w
        ) # Mock reconstructed video tensor
    
    def test_perceptual_loss(self):
        model = PerceptualLoss(
            model_name='vgg16',
            num_frames=2,
            feat_layers=('features.6', 'features.13', 'features.18', 'features.25'),
        )
        
        loss = model(self.rec_video, self.inp_video)
        
        self.assertEqual(loss.shape, torch.Size([]))  # Check the output shape
        self.assertTrue(loss >= 0)
        
    def test_gan_loss(self):
        
        model = GANLoss(
            num_frames=2,
            
            # Discriminator parameters
            frame_size = (self.img_h, self.img_w),
            model_dim = 64,
            dim_mults = (1, 2, 4),
            down_step = (None, 2, 2),
            inp_channels = self.num_channels,
            kernel_size = 3,
            num_groups = 8,
            num_heads = 4,
            dim_head = 32,
        )
        
        
        loss_gen = model(self.rec_video, self.inp_video, train_gen = True)
        loss_dis = model(self.rec_video, self.inp_video, train_gen = False)
        
        self.assertEqual(loss_gen.shape, torch.Size([]))  # Check the output shape
        self.assertEqual(loss_dis.shape, torch.Size([]))  # Check the output shape
        
        self.assertTrue(loss_gen >= 0)
        self.assertTrue(loss_dis >= 0)
        
if __name__ == '__main__':
    unittest.main()
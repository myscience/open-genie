from math import prod
import unittest

import torch
from genie.dynamics import DynamicsModel

class DynamicsModelTestCase(unittest.TestCase):
    def setUp(self):
        
        self.batch_size = 2
        self.video_len = 10
        self.tok_codebook = 16
        self.act_codebook = 4
        self.embed_dim = 64
        self.img_size = 16
        
        TEST_DESC = (
            ('space-time_attn', {
                'n_rep' : 4,
                'n_embd' : self.embed_dim,
                'n_head' : 4,
                'd_head' : 16,
                'transpose' : False,
            }),
        )
        
        self.model = DynamicsModel(
            desc=TEST_DESC,
            tok_vocab=self.tok_codebook,
            act_vocab=self.act_codebook,
            embed_dim=self.embed_dim,
        )
        
        self.mock_tokens = torch.randint(0, self.tok_codebook, (self.batch_size, self.video_len, self.img_size, self.img_size))
        self.mock_act_id = torch.randint(0, self.act_codebook, (self.batch_size, self.video_len))
        
    def test_forward(self):
        # Test the forward method of the DynamicsModel
        logits, last = self.model.forward(self.mock_tokens, self.mock_act_id)
        
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (
            self.batch_size,
            self.video_len,
            self.img_size,
            self.img_size,
            self.tok_codebook,
        ))
        self.assertEqual(last.shape, (
            self.batch_size,
            self.img_size,
            self.img_size,
            self.tok_codebook,
        ))
        
    def test_compute_loss(self):
        # Test the compute_loss method of the DynamicsModel
        loss = self.model.compute_loss(
            self.mock_tokens,
            self.mock_act_id,
        )
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss >= 0)
        
    def test_generate(self):
        # Test the generate method of the DynamicsModel
        generated_frames = self.model.generate(
            self.mock_tokens,
            self.mock_act_id,
            steps=5,
        )
        
        self.assertIsInstance(generated_frames, torch.Tensor)
        self.assertEqual(generated_frames.shape, (
            self.batch_size,
            self.video_len + 1,
            self.img_size,
            self.img_size,
        ))
        
    def test_get_schedule(self):
        # Test the get_schedule method of the DynamicsModel
        steps = 10
        schedule = self.model.get_schedule(steps, (self.img_size, self.img_size))
        
        print(schedule)
        
        self.assertIsInstance(schedule, torch.Tensor)
        self.assertEqual(schedule.shape, (steps,))
        self.assertEqual(torch.sum(schedule), prod((self.img_size, self.img_size)))
        
    def tearDown(self):
        # Clean up any resources used for testing
        pass

if __name__ == '__main__':
    unittest.main()
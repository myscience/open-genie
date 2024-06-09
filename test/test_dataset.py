import unittest

import yaml
from os import path

from genie.dataset import LightningKinetics

# Loading `local_settings.json` for custom local settings
test_folder = path.dirname(path.abspath(__file__))
local_settings = path.join(test_folder, '.local.yaml')

with open(local_settings, 'r') as f:
    local_settings = yaml.safe_load(f)

class TestKineticsDataset(unittest.TestCase):
    def setUp(self):
        self.batch_size = 16
        self.output_format = 'CTHW'
        
        self.dataset = LightningKinetics(
            root=local_settings['kinetics_remote_root'],
            frames_per_clip=16,
            num_classes=local_settings['num_classes'],
            frame_rate=None,
            step_between_clips=1,
            transform=None,
            extensions=('avi', 'mp4'),
            download=local_settings['download'],
            num_download_workers=4,
            num_workers=4,
            output_format=self.output_format,
            batch_size=self.batch_size,
        )

    def test_setup_fit(self):
        self.dataset.setup('fit')
        self.assertIsNotNone(self.dataset.train_dataset)
        self.assertIsNotNone(self.dataset.valid_dataset)
        self.assertIsNone   (self.dataset.test__dataset)

    def test_setup_test(self):
        self.dataset.setup('test')
        self.assertIsNone(self.dataset.train_dataset)
        self.assertIsNone(self.dataset.valid_dataset)
        self.assertIsNotNone(self.dataset.test__dataset)

    def test_setup_invalid_stage(self):
        with self.assertRaises(ValueError):
            self.dataset.setup('invalid_stage')
            
    def test_output_format(self):
        self.assertEqual(self.dataset.output_format, self.output_format)
        
        self.dataset.setup('fit')
        video, lbl = self.dataset.train_dataset[0]
        
        print(video.shape)
        print(lbl)

if __name__ == '__main__':
    unittest.main()
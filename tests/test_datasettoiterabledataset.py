import unittest

import torch
from torch.utils.data import TensorDataset
from hscitorchutil.dataset import DatasetToIterableDataset

class TestDatasetToIterableDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(10))

    def test_toiterabledataset(self):
        ds = DatasetToIterableDataset(self.dataset)
        i = 0
        for item in ds:
            self.assertEqual(item, (i,))
            i += 1
        self.assertEqual(i, 10)

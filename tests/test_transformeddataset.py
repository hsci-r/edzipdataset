import unittest

import torch
from torch.utils.data import TensorDataset
from edzipdataset import TransformedDataset

class TestLinearSubset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(3))

    def test_linearsubset(self):
        ds = TransformedDataset(self.dataset, lambda x: x[0]+2)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[1], 3)
        self.assertEqual(ds[2], 4)
        self.assertEqual(ds.__getitems__([0,2]), [2,4])

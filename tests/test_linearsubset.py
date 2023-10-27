import unittest

import torch
from torch.utils.data import TensorDataset
from edzipdataset import LinearMapSubset

class TestLinearMapSubset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(10))

    def test_linearsubset(self):
        ds = LinearMapSubset(self.dataset, start=3, end=6)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (3,))
        self.assertEqual(ds[1], (4,))
        self.assertEqual(ds[2], (5,))
        self.assertEqual(ds.__getitems__([0,2]), [(3,),(5,)])

import unittest

import torch
from torch.utils.data import TensorDataset
from hscitorchutil.dataset import IdBasedMapSubset

class TestIdBasedMapSubset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(10))

    def test_linearsubset(self):
        ds = IdBasedMapSubset(self.dataset, [3,4,5])
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (3,))
        self.assertEqual(ds[1], (4,))
        self.assertEqual(ds[2], (5,))
        self.assertEqual(ds.__getitems__([0,2]), [(3,),(5,)])

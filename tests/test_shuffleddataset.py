import unittest

import torch
from torch.utils.data import TensorDataset
from edzipdataset import ShuffledMapDataset

class TestShuffledMapDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(3))

    def test_shuffleddataset(self):
        ds = ShuffledMapDataset(self.dataset)
        ds.set_seed(0)
        ds.set_shuffle(True)
        ds.reset()
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (1,))
        self.assertEqual(ds[1], (2,))
        self.assertEqual(ds[2], (0,))
        self.assertEqual(ds.__getitems__([0,2]), [(1,),(0,)])

    def test_disabled_shuffleddataset(self):
        ds = ShuffledMapDataset(self.dataset)
        ds.set_seed(0)
        ds.set_shuffle(False)
        ds.reset()
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (0,))
        self.assertEqual(ds[1], (1,))
        self.assertEqual(ds[2], (2,))
        self.assertEqual(ds.__getitems__([0,2]), [(0,),(2,)])


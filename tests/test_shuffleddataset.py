import unittest

import torch
from torch.utils.data import TensorDataset
from edzipdataset.dsutil import ShuffledMapDataset
import pickle

class TestShuffledMapDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(3))

    def test_shuffleddataset(self):
        ds = ShuffledMapDataset(self.dataset, seed=0)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (2,))
        self.assertEqual(ds[1], (0,))
        self.assertEqual(ds[2], (1,))
        self.assertEqual(ds.__getitems__([0,2]), [(2,),(1,)])


    def test_shuffleddataset_pickling(self):
        ds = ShuffledMapDataset(self.dataset, seed=0)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (2,))
        self.assertEqual(ds[1], (0,))
        self.assertEqual(ds[2], (1,))
        self.assertEqual(ds.__getitems__([0,2]), [(2,),(1,)])
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 3)
        self.assertEqual(ds2[0], (2,))
        self.assertEqual(ds2[1], (0,))
        self.assertEqual(ds2[2], (1,))
        self.assertEqual(ds2.__getitems__([0,2]), [(2,),(1,)])

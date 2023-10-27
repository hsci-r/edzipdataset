import unittest

import torch
from torch.utils.data import TensorDataset
from edzipdataset import TransformedMapDataset
import pickle

class TestTransformedMapDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(3))

    def test_transformedmapdataset(self):
        ds = TransformedMapDataset(self.dataset, lambda x: x[0]+2)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[1], 3)
        self.assertEqual(ds[2], 4)
        self.assertEqual(ds.__getitems__([0,2]), [2,4])

    def test_shuffleddataset_pickling(self):
        ds = TransformedMapDataset(self.dataset, lambda x: x[0]+2)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[1], 3)
        self.assertEqual(ds[2], 4)
        self.assertEqual(ds.__getitems__([0,2]), [2,4])
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 3)
        self.assertEqual(ds2[0], 2)
        self.assertEqual(ds2[1], 3)
        self.assertEqual(ds2[2], 4)
        self.assertEqual(ds2.__getitems__([0,2]), [2,4])

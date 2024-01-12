import unittest

import torch
from torch.utils.data import TensorDataset
from hscitorchutil.dataset import UnionMapDataset, TransformedMapDataset


class TestUnionMapDataset(unittest.TestCase):
    def setUp(self):
        self.datasets = [TensorDataset(torch.arange(3)), TensorDataset(
            torch.arange(3, 6)), TensorDataset(torch.arange(6, 9))]
        self.datasets2 = list(map(lambda d: TransformedMapDataset(
            d, lambda x: x), self.datasets))  # TensorDataset doesn't support __getitems__

    def test_uniondataset(self):
        ds = UnionMapDataset(self.datasets)
        self.assertEqual(len(ds), 9)
        self.assertEqual(ds[0], (0,))
        self.assertEqual(ds[1], (1,))
        self.assertEqual(ds[2], (2,))
        self.assertEqual(ds[3], (3,))
        self.assertEqual(ds[8], (8,))
        self.assertEqual(ds.__getitems__([0, 8, 3, 2]), [
                         (0,), (8,), (3,), (2,)])
        ds = UnionMapDataset(self.datasets2)
        self.assertEqual(ds.__getitems__([0, 8, 3, 2]), [
                         (0,), (8,), (3,), (2,)])

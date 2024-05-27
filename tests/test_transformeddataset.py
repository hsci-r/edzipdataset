import functools
import unittest

import torch
from torch.utils.data import TensorDataset
from hscitorchutil.dataset import TransformedMapDataset
import pickle


def _transform(l):
    return [x[0]+2 for x in l]


def _transform_with_args(l, y):
    return [x[0]+y for x in l]


def _key_transform_with_args(l, y):
    return [x*y for x in l]

class TestTransformedMapDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TensorDataset(torch.arange(3))

    def test_transformedmapdataset(self):
        ds = TransformedMapDataset(self.dataset, lambda l: [x[0]+2 for x in l])
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[1], 3)
        self.assertEqual(ds[2], 4)
        self.assertEqual(ds.__getitems__([0, 2]), [2, 4])

    def test_transformeddataset_pickling(self):
        ds = TransformedMapDataset(self.dataset, _transform)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[1], 3)
        self.assertEqual(ds[2], 4)
        self.assertEqual(ds.__getitems__([0, 2]), [2, 4])
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 3)
        self.assertEqual(ds2[0], 2)
        self.assertEqual(ds2[1], 3)
        self.assertEqual(ds2[2], 4)
        self.assertEqual(ds2.__getitems__([0, 2]), [2, 4])

    def test_transformeddataset_with_args_pickling(self):
        ds = TransformedMapDataset(
            self.dataset, functools.partial(_transform_with_args, y=2), functools.partial(_key_transform_with_args, y=-1))
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], 2)
        self.assertEqual(ds[-1], 3)
        self.assertEqual(ds[-2], 4)
        self.assertEqual(ds.__getitems__([0, -2]), [2, 4])
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 3)
        self.assertEqual(ds2[0], 2)
        self.assertEqual(ds2[-1], 3)
        self.assertEqual(ds2[-2], 4)
        self.assertEqual(ds2.__getitems__([0, -2]), [2, 4])

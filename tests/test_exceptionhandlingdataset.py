from typing import Tuple, Sequence
import unittest

import torch
from torch.utils.data import TensorDataset
from hscitorchutil.dataset import ExceptionHandlingMapDataset, TransformedMapDataset
import pickle


def _except_on_two(tensors: Sequence[Tuple[torch.Tensor, ...]]) -> Sequence[Tuple[torch.Tensor, ...]]:
    if any([tensor == (2,) for tensor in tensors]):
        raise Exception("test")
    else:
        return tensors


class TestExceptionHandlingMapDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = TransformedMapDataset(
            TensorDataset(torch.arange(3)), _except_on_two)

    def test_exceptionhandlingdataset(self):
        ds = ExceptionHandlingMapDataset(self.dataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (0,))
        self.assertEqual(ds[1], (1,))
        self.assertEqual(ds[2], None)
        self.assertEqual(ds.__getitems__([0, 2]), [(0,), None])

    def test_exceptionhandlingdataset_pickling(self):
        ds = ExceptionHandlingMapDataset(self.dataset)
        self.assertEqual(len(ds), 3)
        self.assertEqual(ds[0], (0,))
        self.assertEqual(ds[1], (1,))
        self.assertEqual(ds[2], None)
        self.assertEqual(ds.__getitems__([0, 2]), [(0,), None])
        bytes = pickle.dumps(ds)
        ds2 = pickle.loads(bytes)
        self.assertEqual(len(ds2), 3)
        self.assertEqual(ds2[0], (0,))
        self.assertEqual(ds2[1], (1,))
        self.assertEqual(ds2[2], None)
        self.assertEqual(ds2.__getitems__([0, 2]), [(0,), None])

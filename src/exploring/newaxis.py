# This is an exercise about np.newaxis
# Johannes Siedersleben, QAware GmbH, Munich
# 30/05/2020


import numpy as np
import torch
import unittest
from functools import reduce
from operator import mul

class TestNewaxis(unittest.TestCase):

    def test1(self):
        # newaxis is an alias for None
        self.assertIs(np.newaxis, None)

    def test2(self):
        shape = (2, 2, 3)
        n = reduce(mul, shape)
        a = np.arange(n).reshape(shape)

        k = 0
        b = a[None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 1
        b = a[:, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 2
        b = a[:, :, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 3
        b = a[:, :, :, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

    def test3(self):
        shape = (2, 2, 3)
        n = reduce(mul, shape)
        a = torch.arange(n).reshape(shape)

        k = 0
        b = a[None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 1
        b = a[:, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 2
        b = a[:, :, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

        k = 3
        b = a[:, :, :, None]
        nshape = shape[:k] + (1,) + shape[k:]
        self.assertEqual(nshape, b.shape)

    def test4(self):
        x = torch.arange(12).reshape(4, 3)
        # rows = np.array([0, 3])
        # cols = np.array([0, 2])

        rows = [0, 3]
        cols = [0, 2]

        y = x[rows, cols]
        # y = [x[0, 0], x[3, 2]] = [0, 11]

        z = x[np.ix_(rows, cols)]
        # z = [x[0, 0], x[0, 2],
        #      x[3, 0], x[3, 2]]
        #   = [[0, 2],
        #      [9, 11]]

        print(x)
        print(y)
        print(z)
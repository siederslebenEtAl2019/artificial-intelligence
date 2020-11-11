# This is an exercise about np.newaxis
# Johannes Siedersleben, QAware GmbH, Munich
# 30/05/2020


import numpy as np
import torch
import unittest
from functools import reduce
from operator import mul
import numpy.linalg as LA

class TestNewaxis(unittest.TestCase):

    def test0(self):
        # newaxis is an alias for None
        self.assertIs(np.newaxis, None)

    def test1(self):
        # dividing all rows of a matrix by a vector
        # dividing all columns of a matrix by a vector
        x = np.arange(6).reshape(2,3)
        y = np.array([[4], [5]])
        z = np.array([[2, 3, 4]])

        print(x)
        print(x / y)
        print(x / z)

    def testLA(self):
        x = np.array(([1, 1], [2, 2]))
        nx = np.array([LA.norm(x[i]) for i in range(x.shape[0])]).reshape(-1, 1)
        print(nx)

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
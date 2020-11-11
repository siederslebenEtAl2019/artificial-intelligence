import unittest

from bayes.util.binaries import *
from bayes.util.digraphs import *


class TestUtil(unittest.TestCase):
    def test_binary_cube(self):
        c = list(binary_cube(()))
        c = list(binary_cube((0,)))
        c = list(binary_cube((None,)))
        c = list(binary_cube((0, 0)))
        c = list(binary_cube((0, None)))
        c = list(binary_cube((None, 0)))
        c = list(binary_cube((None, None)))
        c = list(binary_cube((0, 1, None, 0, 1, None, 0, 1, None)))
        print(c)

    def test_intbin(self):
        def id1(b):
            return int2bin((bin2int(b)), len(b))

        def id2(i):
            return bin2int(int2bin(i, log2(i)))

        for i in range(1000):
            self.assertEqual(i, id2(i))

        for i in range(2 ** 10):
            b = int2bin(i, log2(i))
            self.assertEqual(b, id1(b))

    def test_rev_int(self):
        for i in range(100000):
            j = rev_int(i, 20)
            k = rev_int(j, 20)
            self.assertEqual(i, k)

    def test_bin2int(self):
        for i in range(100000):
            b = int2bin(i, 15)
            j = bin2int(b)
            self.assertEqual(i, j)

    def testBackprop(self):
        bst = ()
        grad = backprop(bst, 0)
        print(grad)



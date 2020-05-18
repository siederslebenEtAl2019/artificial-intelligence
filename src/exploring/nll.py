import unittest

import torch
import torch.nn as nn
import math


def crossent(x, i):
    return -x[i] + math.log(sum([math.exp(a) for a in x]))




class MyTestCase(unittest.TestCase):
    def test_nll(self):
        print()
        m = nn.Softmax(dim=1)

        a = torch.tensor([[1., 1., 1.]])
        print(m(a))
        a = torch.tensor([[0., 0., 0.]])
        print(m(a))
        a = torch.tensor([[1., 0., 0.]])
        print(m(a))
        a = torch.tensor([[0., 1., 0.]])
        print(m(a))

    def test_crossentropy(self):
        print()
        m = nn.CrossEntropyLoss()
        t = 0
        tt = torch.tensor([t])     # 0 = true value

        xs = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1/3, 1/3., 1/3]]
        for x in xs:
            tx = torch.tensor([x])
            s = sum(m(tx, torch.tensor([t])).item() for t in range(3))
            print(m(tx, tt).item(), crossent(x, t), s)

    def testMinibatch(self):
        print()
        m = nn.CrossEntropyLoss()
        tt = torch.tensor([0, 2])           # list of true values
        xs = [[1., 0., 0.], [1., 0., 0.]]   # list of densities

        c1 = crossent([1., 0., 0.], 0)
        c2 = crossent([1., 0., 0.], 2)
        avg = 0.5 * (c1 + c2)

        tx = torch.tensor(xs)
        print(m(tx, tt), avg)







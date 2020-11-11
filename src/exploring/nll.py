import unittest
import torch
import torch.nn as nn


def exp_sum(z):
    return torch.exp(z).sum()


def mysoftmax(z):
    return torch.exp(z) / exp_sum(z)


class MyTestCase(unittest.TestCase):
    def testSoftmax(self):
        # this testcase shows:
        # -log(softmax(z)) = -z + log(sum(exp(z))

        print()
        sm = nn.Softmax(dim=1)

        z = torch.tensor([[5., 7., 9.]])
        t = torch.log(sm(z)) + torch.log(exp_sum(z))
        self.assertTrue((z.allclose(t)))


    def testCrossentropy1(self):
        """
        This testcase shows:

        """
        sm = nn.Softmax(dim=1)
        loss = nn.CrossEntropyLoss()
        y = torch.tensor([2])  # label 2 is true
        z = torch.tensor([[5, 7, 9]], dtype=torch.float)  # input

        # three ways to compute cross entropy loss
        loss0 = loss(z, y)
        loss1 = -torch.log(sm(z))[0, y[0]]
        loss2 = -z[0, y[0]] + torch.log(exp_sum(z))
        self.assertTrue(loss0.isclose(loss1))
        self.assertTrue(loss0.isclose(loss2))


    def testCrossentropy2(self):
        sm = nn.Softmax(dim=1)
        loss = nn.CrossEntropyLoss()
        y = torch.tensor([0, 2])  # vector of true values
        z = torch.tensor([[5, 7, 9], [6, 8, 10]], dtype=torch.float)  # input

        a = z[0, None]
        b = z[1, None]

        u = y[0, None]
        v = y[1, None]

        dist = loss(z, y)
        dist0 = loss(z[0, None], y[0, None])
        dist1 = loss(z[1, None], y[1, None])
        self.assertTrue(dist.isclose((dist0 + dist1) / 2))

        dist0 = -torch.log(sm(z))[0, y[0]]
        dist1 = -torch.log(sm(z))[0, y[1]]
        self.assertTrue(dist.isclose((dist0 + dist1) / 2))

        dist0 = -z[0, y[0]] + torch.log(exp_sum(z[0, None]))
        dist1 = -z[1, y[1]] + torch.log(exp_sum(z[1, None]))
        self.assertTrue(dist.isclose((dist0 + dist1) / 2))






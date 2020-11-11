import unittest

import torch
import torch.nn as nn

class TestLinear(unittest.TestCase):

    def testMM(self):
        n = 4  # number of measurements
        p = 2  # number of input features
        q = 3  # number of output features

        b_shape = [q]
        w_shape = [q, p]
        x_shape = [n, p]

        b = torch.tensor([3., 3., 3.])  # bias
        b = b.view(b_shape)

        w = torch.tensor([1., 1., 1., 1., 1., 1.])  # weight
        w = w.view(w_shape)

        x = torch.tensor([11., 12., 21., 22., 31., 32., 41., 42.])  # features
        x = x.view(x_shape)

        # mm stands for matrix multiplication
        # b is automatically added to all rows
        # two ways of multiplying matrices
        y = x.mm(w.t()) + b
        z = w.mm(x.t()).t() + b
        self.assertTrue(torch.equal(y, z))

    def testStateDict(self):
        n = 4  # number of measurements
        p = 2  # number of input features
        q = 3  # number of output features

        l_shape = [p, q]
        x_shape = [n, p]

        # lin is a function object as described above
        lin = nn.Linear(*l_shape)  # weight : p x q, bias : q

        # getting weight and bias from the state dictionary
        dict = lin.state_dict()
        w = dict['weight']
        b = dict['bias']

        x = torch.tensor([11., 12., 21., 22., 31., 32., 41., 42.])  # features
        x = x.view(x_shape)

        # applying lin = Linear(2, 3) and comparing to mm
        y = lin(x)
        z = x.mm(w.t()) + b  # (n x p) * (p x q)  + (n x q)

        self.assertTrue(torch.equal(y, z))

    def testBackward(self):
        n = 4  # number of measurements
        p = 2  # number of input features
        q = 3  # number of output features

        l_shape = [p, q]
        x_shape = [n, p]

        x = torch.tensor([11., 12., 21., 22., 31., 32., 41., 42.])  # features
        x = x.view(x_shape)

        # lin is a function object as described above
        lin = nn.Linear(*l_shape)  # weight : p x q, bias : q

        dict = lin.state_dict()
        w = dict['weight']
        b = dict['bias']

        # applying lin = Linear(2, 3) and checking the result
        y = lin(x)

        e1 = torch.ones((n, q))


        # computing the gradients for weight and bias with respect to s
        y.backward(e1)

        # one step of gradient descent applied to weight and bias
        for p in lin.parameters():
            p.data -= p.grad.data

        # getting weight and bias from the state dictionary
        xd = lin.state_dict()
        w = xd['weight']
        b = xd['bias']

        print(lin.state_dict()['weight'])
        print(lin.state_dict()['bias'])


        # applying lin = Linear(2, 3) with modified parameters and checking the result
        y = lin(x)
        if y is None:
            raise Exception

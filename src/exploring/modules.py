import unittest
import torch
import torch.nn as nn


class TestModules(unittest.TestCase):

    def testLinear(self):
        """
        Linear works as follows:
        The constructor expects two arguments:
        p = size of input and q = size of output.
        The input vector x is (p x n), n = number of measurements (n >= 1)
        Linear works for any n >= 1.
        It generates a matrix w = weight : q x n and a vector b = bias : q
        Linear returns a function object lin which computes

        lin(x) = x.mm(w.t()) + b : q x n,

        b being added to each row (there are n rows of length q)
        "weight" and "bias" are called the parameters of Linear.
        They are stored in Linear's state dictionary (state_dict)
        This example also shows how parameters can be viewed and modified.
        """

        w_shape = [3, 2]
        b_shape = [3]
        x_shape = [4, 2]

        # lin is a function object as described above
        lin = nn.Linear(2, 3)     # weight : 3 x 2, bias : 3

        # x is an input vector with two features (per row) four times measured
        x = torch.tensor([5., 6., 30., 40., 200., 600., 0., 0],
                         requires_grad=True).reshape(x_shape)  # x : 4 x 2

        # setting weight and bias to arbitrary values
        w = torch.full(w_shape, 2.)
        b = torch.full(b_shape, 7.)
        xd = {'weight': w, 'bias': b}
        lin.load_state_dict(xd)

        # applying lin = Linear(2, 3) and checking the result
        y = lin(x)
        z = x.mm(w.t()) + b  # (3 x 2) * (2 x 1)  + (3 x 1)
        self.assertTrue(torch.equal(y, z))

        # sum is an arbitrary scalar-valued function
        s = y.sum()

        # computing the gradients for weight and bias with respect to s
        s.backward()

        # one step of gradient descent applied to weight and bias
        for p in lin.parameters():
            p.data -= p.grad.data

        # getting weight and bias from the state dictionary
        xd = lin.state_dict()
        w = xd['weight']
        b = xd['bias']

        # applying lin = Linear(2, 3) with modified parameters and checking the result
        y = lin(x)
        z = x.mm(w.t()) + b
        self.assertTrue(torch.equal(y, z))





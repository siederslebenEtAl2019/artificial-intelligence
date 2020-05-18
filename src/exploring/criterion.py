import unittest
import torch
import torch.nn as nn


class TestCriterions(unittest.TestCase):
    def test_nllloss0(self):

        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss()

        # each element in target has to have 0 <= value < C
        # target[i] = target value of i-th row of input
        # input is of size N x C = 1 x n

        n = 10
        data = torch.zeros((1, n), dtype=torch.float)
        data[0, -1] = 1
        input = m(data)

        print('\n', data)
        print(input)

        # bad match for i = 0, i = 1, good match for i = n - 1
        for i in (0, 1, n-1):
            target = torch.tensor([i])   # 0 <= i < n
            output = loss(input, target)
            print(target.item(), output.item())

    def test_nllloss1(self):
        m = nn.LogSoftmax(dim=1)
        loss = nn.NLLLoss()

        n = 10
        data = torch.ones((1, n), dtype=torch.float)
        input = m(data)

        print('\n', data)
        print(input)

        # bad match for i = 0, i = 1, i = n - 1
        for i in (0, 1, n - 1):
            target = torch.tensor([i])  # 0 <= i < n
            output = loss(input, target)
            print(target.item(), output.item())



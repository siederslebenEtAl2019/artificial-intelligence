# Johannes Siedersleben, QAware GmbH, Munich, Germany
# 18/05/2020

# Exploring autograd

import unittest

import torch
import torch.nn as nn


class Square(nn.Module):
    def __init__(self, start):
        super(Square, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.tensor(start))])

    def forward(self):
        return self.params[0] ** 2


class TestOptimizing(unittest.TestCase):

    def testA(self):
        """
        min x**2, no model
        """
        n_iterations = 100
        epsilon = 1e-7
        eta = 1e-1
        start = 1.
        target = 0
        protocol = []

        x = torch.tensor(start, requires_grad=True)
        y = torch.tensor(target)
        loss = nn.L1Loss()
        optimizer = torch.optim.SGD([x], lr=eta)

        while loss(x ** 2, y) > epsilon:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            protocol.append([loss.item(), (x ** 2).data.item(), x.item()])

        for record in protocol:
            print(f'{record[0] + 1:4} {record[1]:15.10f}'
                  f'{record[2]:15.10f}'
                  f'{record[3]:15.10f}')

    def testB(self):
        """
           min x.norm(), no model
           """
        n_epochs = 100
        epsilon = 1e-6
        eta = 1e-1
        start = [1., 1.]
        target = 0
        protocol = []

        x = torch.tensor(start, requires_grad=True)
        y = torch.tensor(target)
        criterion = nn.L1Loss()
        optimizer = torch.optim.Adam([x], lr=eta)

        for epoch in range(n_epochs):
            current = torch.norm(x)
            loss = criterion(current, y)
            if abs(loss) < epsilon:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            protocol.append([epoch + 1, loss.item(), current.data.item(), x.data[0].item(), x.data[1].item()])

        for record in protocol:
            print(f'{record[0] + 1:4} {record[1]:15.10f}'
                  f'{record[2]:15.10f}'
                  f'{record[3]:15.10f}'
                  f'{record[4]:15.10f}')

    def testC(self):
        """
             min x**2, model = Square
        """
        n_epochs = 100
        epsilon = 1e-7
        eta = 1e-1
        start = 1.
        target = 0.
        protocol = []

        model = Square(start)
        y = torch.tensor(target)
        criterion = nn.L1Loss()
        optimizer = torch.optim.SGD(model.parameters(), lr=eta)

        for epoch in range(n_epochs):
            current = model()
            loss = criterion(current, y)
            if abs(loss) < epsilon:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            protocol.append([epoch + 1, loss.item(), current.data.item(),
                             next(model.parameters()).item()])

        for record in protocol:
            print(f'{record[0] + 1:4} {record[1]:15.10f}'
                  f'{record[2]:15.10f}'
                  f'{record[3]:15.10f}')

    def testD(self):
        """
           min mse(x.dot(w))
           """
        n_epochs = 201
        epsilon = 1e-5
        eta = 1e-1
        start = [[1.], [2.], [3.]]
        target = [[1.], [2.], [3.]]
        protocol = []
        every_nth = 10

        m = 3  # number of measurements
        n = 1  # number of input features
        K = 1  # number of output features

        lin = nn.Linear(n, K)  # weight : n x K, bias : K
        lin.weight = nn.Parameter(torch.tensor([[2.]], requires_grad=True))
        lin.bias = nn.Parameter(torch.tensor([1.], requires_grad=True))

        x = torch.tensor(start, requires_grad=True)
        y = torch.tensor(target)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lin.parameters(), lr=eta)

        for epoch in range(n_epochs):
            current = lin(x)
            loss = criterion(current, y)
            if abs(loss) < epsilon:
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curr = [c.data.item() for c in current]
            weight = [w.data.item() for w in lin.weight]
            bias = [b.data.item() for b in lin.bias]
            protocol.append([epoch, loss.item(), curr, weight, bias])

        for counter, record in enumerate(protocol):
            if counter % every_nth == 0:
                print(f'{record[0]:4} {record[1]:15.10f}', end='')
                for c in record[2]:
                    print(f'{c:15.10f}', end='')
                for w in record[3]:
                    print(f'{w:15.10f}', end='')
                for b in record[4]:
                    print(f'{b:15.10f}', end='')
                print()







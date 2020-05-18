import unittest

import torch
import torch.nn as nn

# device = torch.device("cuda")
device = torch.device("cpu")


def printNode(x):
    print('\n', x.data, '\n', x.grad, '\n', x.grad_fn)


class PNN(nn.Module):  # plain neural network
    def __init__(self, factor):
        super(PNN, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * x * self.factor


class RNN(nn.Module):  # recurrent neural network
    def __init__(self):
        super(RNN, self).__init__()

    def forward(self, input, hidden):
        output = input * 2 + hidden
        return output


class Testrecurrent(unittest.TestCase):

    def test1(self):
        device = torch.device("cpu")

        i = 0
        pnn = PNN(1)
        criterion = nn.MSELoss(reduction='sum')
        target = torch.tensor(1., device=device)
        learning_rate = 1e-2
        loss = torch.tensor(1000., device=device)

        x = torch.tensor(10., requires_grad=True, device=device)
        optimizer = torch.optim.Adam([x], lr=learning_rate)

        for i in range(10000):
            pnn.zero_grad()  # set gradients to zero
            y = pnn(x)  # update y
            loss = criterion(y, target)  # calculate loss
            if loss.item() < 1e-6:
                break
            loss.backward()  # calculate gradients
            optimizer.step()  # perform optimization

        print('\n', f'{i + 1:6} {loss.item():8.6f} {x.data.item():8.6f} {x.grad.item():8.6f}')

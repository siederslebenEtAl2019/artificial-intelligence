# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy
# neural network baseline torch

import torch


def euclid(x, t):
    return torch.dist(x, t)


def deuclid(x, t):
    return 2 * (x - t)


def identity(x):
    return x


def didentity(x):
    return 1.


def relu(x):
    return x.clamp(min=0)


def drelu(x):
    return (torch.sign(x) + 1) / 2


sigmoid = torch.sigmoid


def dsigmoid(x):  # so is this
    return sigmoid(x) * (1 - sigmoid(x))


act_functions = {'relu': relu,
                 'drelu': drelu,
                 'sigmoid': sigmoid,
                 'dsigmoid': dsigmoid,
                 'identity': identity,
                 'didentity': didentity}


def solve(x, t):
    """
    :param x: input vector (p x n)
    :param t: target vector (q x n)
    :return: optimal w, minimal distance between w.T.dot(x) and t; (p x q),
    """
    w = torch.inverse(x.mm(x.t())).mm(x).dot(t())
    return w, euclid(w.t().mm(x), t)


def backprop(phi, dphi, cost, dcost, eta, x0, w, control):
    """
    :param phi: array of activation functions, starts at 1
    :param dphi: array of derivatives of phi, starts at 1
    :param cost: cost function, usually euclid(x, t)
    :param dcost: derivative of cost function
    :param eta: iterator yielding stepsize
    :param x0: input matrix p[0] x n
    :param w: starting values of w[1] to w[m] : p[k-1] x p[k]
    :param control: takes care of terminating condition and verbosity
    :return: solution w, control containing protocol

    The other variables are:
    w: array of weight matrices; starts at 1
    dw: array of gradients of w; starts at 1
    x: array of value matrices; x[k]= phi[k](y[k]); starts at 0
    y: array of intermediate values: y[k] = w[k].T.dot(x[k-1]
    """

    # w, phi and dphi must be of equal length
    if len(w) != len(phi) or len(w) != len(dphi):
        raise ValueError

    # x0.shape[0] = w[1].shape[0] = number of input variables
    if x0.shape[0] != w[1].shape[0]:
        raise ValueError

    m = len(w) - 1  # number of layers

    x = [x0]  # starts at 0
    y = [None]  # starts at 1

    # first forward propagation: allocate dw, compute y[k], x[k]
    for k in range(1, m + 1):
        x.append(phi[k](w[k].t().mm(x[k - 1])))

    current = cost(x[m])
    control.write(current)

    # main trainloop
    while control.carryon():
        current.backward()

        # forward propagation: update w[k], recompute y[k], x[k]
        mue = next(eta)
        with torch.no_grad():
            for k in range(1, m + 1):
                w[k] -= mue * w[k].grad
                w[k].grad.zero_()

        for k in range(1, m + 1):
            x[k] = phi[k](w[k].t().mm(x[k - 1]))

        current = cost(x[m])
        control.write(current)

    return [None] + [v.detach().numpy() for v in w[1:]], control

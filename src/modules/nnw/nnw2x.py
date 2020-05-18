# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import numpy as np


def backprop(phi, dphi, cost, dcost, eta, x0, w, control):
    """
    :param phi: array of activation functions, starts at 1
    :param dphi: array of derivatives of phi, starts at 1
    :param cost: cost function, usually euclid(x, t)
    :param dcost: derivative of cost function
    :param ieta: iterator yielding stepsize
    :param x0: input matrix p[0] x n
    :param w: starting values of w[1] to w[m] : p[k-1] x p[k]
    :param control: takes care of terminating condition and verbosity
    :return: solution w, minimal cost

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
    if m != 2:
        raise ValueError

    dw = []
    y = []
    x = []

    # first forward propagation: allocate dw, compute y[k], x[k]
    for k in range(0, 3):
        dw.append(None)
        y.append(None)
        x.append(None)

    x[0] = x0
    y[1] = w[1].T.dot(x[0])
    x[1] = np.maximum(y[1], 0)
    y[2] = w[2].T.dot(x[1])
    x[2] = y[2]

    control.write(cost(x[2]))

    # main trainloop
    while control.carryon():
        delta = dcost(x[2])  # dphi[2] = 1
        dw[2] = x[1].dot(delta.T)

        delta = w[2].dot(delta)
        delta[y[1] < 0] = 0  # dphi[1] = heavyside
        dw[1] = x[0].dot(delta.T)

        # forward propagation: update w[k], recompute y[k], x[k]
        w[1] -= next(eta) * dw[1]
        y[1] = w[1].T.dot(x[0])
        x[1] = np.maximum(y[1], 0)  # phi[1] = relu

        w[2] -= next(eta) * dw[2]
        y[2] = w[2].T.dot(x[1])
        x[2] = y[2]  # phi[2] = identity

        control.write(cost(x[2]))

    return w, control

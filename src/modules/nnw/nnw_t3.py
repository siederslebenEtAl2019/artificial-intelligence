# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import torch


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

    dw = [None]  # starts at 1
    x = [x0]  # starts at 0
    y = [None]  # starts at 1

    # first forward propagation: allocate dw, compute y[k], x[k]
    for k in range(1, m + 1):
        dw.append(None)
        y.append(w[k].t().mm(x[k - 1]))
        x.append(phi[k](y[k]))

    loss = cost(x[m])
    loss.backward()
    control.write(loss)

    # main trainloop
    mue = next(eta)
    while control.carryon():
        with torch.no_grad():
            for k in range(1, m):
                w[k] -= mue * w[k].grad
                y[k] = w[k].t().mm(x[k - 1])
                x[k] = phi[k](y[k])
        loss = cost(x[m])
        loss.backward()
        control.write(loss)

    return [None] + [v.numpy() for v in w[1:]], control

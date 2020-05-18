# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import torch

dtype = torch.float
# device = torch.device("cpu")
device = torch.device("cuda:0")  # Uncomment this to run on GPU


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
    :return: solution w, minimal cost

    The other variables are:
    w: array of weight matrices; starts at 1
    dw: array of gradients of w; starts at 1
    x: array of value matrices; x[k]= phi[k](y[k]); starts at 0
    y: array of intermediate values: y[k] = w[k].T.dot(x[k-1]
    """

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    # N, D_in, H, D_out = 64, 1000, 100, 10 # n, p0, p1, p2

    # Create random input and output data
    # x = torch.randn(N, D_in, device=device, dtype=dtype)
    # y = torch.randn(N, D_out, device=device, dtype=dtype)

    m = len(w) - 1  # number of layers
    if m != 2:
        raise ValueError

    x_ = x0.t()

    # Randomly initialize weights
    # w1 = torch.randn(D_in, H, device=device, dtype=dtype)      # p[0] x p[1]
    # w2 = torch.randn(H, D_out, device=device, dtype=dtype)     # p[1] x p[2]

    w1 = w[1]
    w2 = w[2]
    w_ = [None, w1, w2]

    # main trainloop
    while control.carryon():
        # Forward pass: compute predicted y
        h = x_.mm(w1)  # y[1] = w1.dot(x)
        h_relu = h.clamp(min=0)  # x[1] = relu(y[0])
        y_pred = h_relu.mm(w2)  # y[2] = w2.dot(x[1], x[2] = identity(y[2])

        # Compute and print loss
        loss = cost(y_pred.t())  # loss = (y_pred - y).pow(2).sum().item()
        # loss = euclid(x[2], t)
        # print(t, loss)
        control.write(loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        # grad_y_pred = 2.0 * (y_pred - y)    # delta = dcost(x[2]) * dphi[2](y[2])
        grad_y_pred = dcost(y_pred.t()).t()  # delta = dcost(x[2]) * dphi[2](y[2])
        grad_w2 = h_relu.t().mm(grad_y_pred)  # dw[2] = x[1].T.dot(delta)

        grad_h_relu = grad_y_pred.mm(w2.t())  # delta = delta.dot(w[2])
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0  # delta = delta * dphi[1](y[1])
        grad_w1 = x_.t().mm(grad_h)  # dw[1] = x[0].T.dot(delta)

        # Update weights
        w1 -= next(eta) * grad_w1  # w[1] -= eta * dw[1]
        w2 -= next(eta) * grad_w2  # w[2] -= eta * dw[2]

    return [None] + [v.numpy() for v in w_[1:]], control

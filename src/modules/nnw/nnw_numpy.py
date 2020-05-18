# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy
# neural network baseline numpy

from collections import deque

import numpy as np


class Control(object):
    def __init__(self, name, span, verbosity, precision, steps):
        """
        :param name: name of testcase
        :param span: length of control interval
        :param verbosity: write report on every verbosity-th step
        :param precision: stop if span of control interval is less than precision
        :param steps: max number of steps
        """
        self.name = name
        self.span = span
        self.verbosity = verbosity
        self.precision = precision
        self.steps = steps
        self.counter = 0
        self.report = []
        self.memory = deque(maxlen=span)

    def copy(self):
        return Control(self.name, self.span,
                       self.verbosity, self.precision, self.steps)

    def write(self, value):
        if self.verbosity > 0 and self.counter % self.verbosity == 0:
            self.report.append((self.counter, value))
        self.counter += 1
        self.memory.append(value)

    def carryon(self):
        return self.counter < self.steps \
               and (len(self.memory) < self.span
                    or max(self.memory) - min(self.memory)) > self.precision

    def lastvalue(self):
        return self.memory[-1]


def etaiterator(start, hvperiod):
    """
    :param start: start value
    :param hvperiod: value halves after that many iterations
    :return: iterator yielding decreasing values
    """

    def iterator(s=start, h=hvperiod):
        factor = 1 if h == 0 else 2 ** (-1 / h)
        while True:
            yield s
            s *= factor

    return iterator


def euclid(x, t):
    return ((x - t) ** 2).sum()


def deuclid(x, t):
    return 2 * (x - t)


def identity(x):
    return x


def didentity(x):
    return 1.


def relu(x):
    return np.maximum(x, 0)


def drelu(x):
    return np.heaviside(x, 0)


def sigmoid(x):  # this is a ufunc
    return 1 / (1 + np.exp(-x))


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
    w = np.linalg.inv(x.dot(x.T)).dot(x).dot(t.T)
    return w, euclid(w.T.dot(x), t)


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
        y.append(w[k].T.dot(x[k - 1]))
        x.append(phi[k](y[k]))

    control.write(cost(x[m]))

    # main trainloop
    while control.carryon():
        delta = dcost(x[m]) * (dphi[m](y[m]))
        dw[m] = x[m - 1].dot(delta.T)

        # backward propagation: compute dw[k]
        for k in range(m - 1, 0, -1):
            delta = w[k + 1].dot(delta) * dphi[k](y[k])
            dw[k] = x[k - 1].dot(delta.T)

        # forward propagation: update w[k], recompute y[k], x[k]
        mue = next(eta)
        for k in range(1, m + 1):
            w[k] -= mue * dw[k]
            y[k] = w[k].T.dot(x[k - 1])
            x[k] = phi[k](y[k])

        control.write(cost(x[m]))

    return w, control

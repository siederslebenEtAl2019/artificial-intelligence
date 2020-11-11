# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import pickle
import numpy as np
import torch

from modules.nnw import nnw_numpy, nnw_torch

datapath = "C:/Users/j.siedersleben/PycharmProjects/akki-pytorch/test/data/"
dtype = torch.float
# device = torch.device("cuda:0")
device = torch.device("cpu")


class Testmanager(object):
    def __init__(self, name, phi, dphi, eta, span, verbosity, precision, steps):
        """
        :param name: name of testcase
        :param phi: array of activation functions, starts at 1
        :param dphi: array of derivatives of phi, starts at 1
        :param eta: iterator yielding stepsize
        :param span: length of control interval
        :param verbosity: write report on every verbosity-th step
        :param precision: stop if span of control interval is less than precision
        :param steps: max number of steps
        """
        name, m, n, p, t, w, x0 = readTestdata(name)
        self.name = name
        self.eta = eta
        self.phi = phi.copy()
        self.dphi = dphi.copy()
        self.m = m  # number of layers
        self.n = n  # number of measurements
        self.p = p  # p[k] = height of k-th layer
        self.t = t if t is not None else random_t(m, p)
        self.w = w if w is not None else random_w(m, p)
        self.x0 = x0 if x0 is not None else random_x0(p[0], n)
        self.control = nnw_numpy.Control(name, span, verbosity, precision, steps)

        if len(phi) != self.m or len(dphi) != self.m:
            raise ValueError

    def random_w(self):
        self.w = random_w(self.m, self.p)

    def getArgsNumpy(self):
        return {'phi': [None] + [nnw_numpy.act_functions[q] for q in self.phi],
                'dphi': [None] + [nnw_numpy.act_functions[q] for q in self.dphi],
                'eta': self.eta(),
                'cost': lambda x: nnw_numpy.euclid(x, self.t),
                'dcost': lambda x: nnw_numpy.deuclid(x, self.t),
                'w': copy_w(self.w),
                'x0': self.x0,
                'control': self.control.copy()}

    def getArgsTorch(self):
        t = torch.tensor(self.t, dtype=dtype, device=device)
        return {'phi': [None] + [nnw_torch.act_functions[q] for q in self.phi],
                'dphi': [None] + [nnw_torch.act_functions[q] for q in self.dphi],
                'eta': self.eta(),
                'cost': lambda x: nnw_torch.euclid(x, t),
                'dcost': lambda x: nnw_torch.deuclid(x, t),
                'w': torch_w(self.w),
                'x0': torch.tensor(self.x0, dtype=dtype, device=device),
                'control': self.control.copy()}


def copy_w(w):
    return [None] + [v.copy() for v in w[1:]]


def torch_w(w):
    return [None] + [torch.tensor(v,
                                  dtype= dtype,
                                  device=device,
                                  requires_grad=True) for v in w[1:]]

def numpy_w(w):
    return [None] + [v.numpy() for v in w[1:]]


def random_t(pm, n):
    return np.random.randn(pm, n)


def random_w(m, p):
    w = [None]
    for k in range(1, m + 1):
        w.append(np.random.randn(p[k - 1], p[k]))
    return w


def random_x0(p0, n):
    return np.random.randn(p0, n)


def storeTestdata(name, m, n, p, t, w, x0):
    data = [name, m, n, p, t, w, x0]
    with open(datapath + name + '.pickle', 'wb') as handle:
        pickle.dump(data, handle)


def readTestdata(name):
    with open(datapath + name + '.pickle', 'rb') as handle:
        return pickle.load(handle)

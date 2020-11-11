# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy


import numpy as np

from modules.nnw import nnw_numpy
from util.testutil import Testmanager, random_t, random_w, random_x0, storeTestdata


def storeSimple0():
    name, m, n, p = 'simple0', 2, 1, [1, 1, 1]
    t = np.array([[1.]])
    w = [None, np.array([[1.]]), np.array([[1.]])]
    x0 = np.array([[-1.]])
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def simple0():
    return Testmanager('simple0',
                       ['identity', 'identity'], ['didentity', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-3, 0),
                       span=2, precision=1e-10, verbosity=1000, steps=30000)


def storeSimple1():
    name, m, n, p = 'simple2', 2, 100, [10, 10, 10]
    t = random_t(p[m], n)
    w = random_w(m, p)
    x0 = random_x0(p[0], n)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def simple1():
    return Testmanager('simple2',
                       ['relu', 'relu'], ['drelu', 'drelu'],
                       eta=nnw_numpy.etaiterator(1e-4, 0),
                       span=2, precision=1e-8, verbosity=1000, steps=30000)


def storeSimple2():
    name, m, n, p = 'simple3', 2, 15, [10, 10, 5]
    t = random_t(p[m], n)
    w = random_w(m, p)
    x0 = random_x0(p[0], n)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def simple2():
    return Testmanager('simple3',
                       ['sigmoid', 'identity'], ['dsigmoid', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-4, 10000),
                       span=10, precision=1e-8, verbosity=10000, steps=200000)


def storeSimple3():
    name, m, n, p = 'simple4', 3, 30, [10, 10, 10, 10]
    t = np.full(p[m], 17.)
    w = [None, np.full((p[0], p[1]), 3.), np.full((p[1], p[2]), 3.), np.full((p[2], p[3]), 3.)]
    x0 = np.full(p[0], 17.)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def simple3():
    return Testmanager('simple4',
                       [nnw_numpy.relu, nnw_numpy.identity, nnw_numpy.identity], [nnw_numpy.drelu, nnw_numpy.didentity, nnw_numpy.didentity],
                       eta=nnw_numpy.etaiterator(1e-3, 0),
                       span=10, precision=1e-10, verbosity=100, steps=30000)


def storeAll(s):
    for t in [storeSimple0, storeSimple1, storeSimple2, storeSimple3][s]:
        t()


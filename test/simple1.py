# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy


import numpy as np

from modules.nnw import nnw_numpy
from util.testutil import Testmanager, random_t, random_w, random_x0, storeTestdata


def storeSimple10():
    name, m, n, p = 'simple10', 2, 1, [1, 1, 1]
    t = np.array([[1.]])
    w = [None, np.array([[1.]]), np.array([[1.]])]
    x0 = np.array([[1.]])
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def storeSimple11():
    name, m, n, p = 'simple11', 2, 1, [1, 1, 1]
    t = np.array([[1.]])
    w = [None, np.array([[0.]]), np.array([[0.]])]
    x0 = np.array([[1.]])
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def storeSimple12():
    name, m, n, p = 'simple12', 2, 3, [2, 2, 2]
    t = np.array([[1.], [1.]])
    w = [None, np.full((2, 2), 3.), np.full((2, 2), 3.)]
    x0 = np.array([[1.], [1.]])
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def storeSimple13():
    name, m, n, p = 'simple13', 2, 64, [1000, 100, 10]
    t = random_t(p[m], n)
    w = random_w(m, p)
    x0 = random_x0(p[0], n)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def simple10():
    return Testmanager('simple10',
                       ['relu', 'identity'], ['drelu', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-4, 0),
                       span=2, precision=1e-10, verbosity=1000, steps=30000)


def simple11():
    return Testmanager('simple11',
                       ['relu', 'identity'], ['drelu', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-4, 0),
                       span=2, precision=1e-10, verbosity=1000, steps=30000)


def simple12():
    return Testmanager('simple12',
                       ['relu', 'identity'], ['drelu', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-4, 0),
                       span=2, precision=1e-10, verbosity=1000, steps=30000)


def simple13():
    return Testmanager('simple13',
                       ['relu', 'identity'], ['drelu', 'didentity'],
                       eta=nnw_numpy.etaiterator(1e-5, 0),
                       span=5, precision=1e-10, verbosity=200, steps=4000)


def storeAll(s):
    for t in [storeSimple10, storeSimple11, storeSimple12, storeSimple13][s]:
        t()


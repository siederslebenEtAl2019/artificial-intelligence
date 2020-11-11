# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import numpy as np

from modules.nnw import nnw_numpy
from util.testutil import Testmanager, random_w, storeTestdata


def storeTiny0():
    name, m, n, p = 'tiny0', 1, 1, [1, 1]
    t = np.array([[1.]])
    x0 = np.array([[1.]])
    w = [None, np.array([[-1.]])]
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def tiny0():
    return Testmanager('tiny0',
                       ['identity'], ['didentity'],
                       eta=nnw_numpy.etaiterator(1e-3, 0),
                       span=10, precision=1e-8, verbosity=2000, steps=100000)


def storeTiny1():
    name, m, n, p = 'tiny1', 1, 3, [1, 1]
    t = np.array([[1, 2, 3]])
    x0 = np.array([[3, 7, 4]])
    w = random_w(m, p)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def tiny1():
    return Testmanager('tiny1',
                       ['identity'], ['didentity'],
                       eta=nnw_numpy.etaiterator(5e-4, 0),
                       span=10, precision=1e-8, verbosity=1000, steps=5000)


def storeTiny2():
    name, m, n, p = 'tiny2', 1, 8, [2, 2]
    t = np.array([[3, 4, 5, 6, 7, 8, 9, 10], [4, 6, 8, 9, 10, 12, 14, 15]])
    x0 = np.array([[3, 5, 6, 8, 8, 9, 10, 12], [3, 5, 8, 4, 9, 14, 12, 16]])
    w = random_w(m, p)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def tiny2():
    return Testmanager('tiny2',
                       ['identity'], ['didentity'],
                       eta=nnw_numpy.etaiterator(5e-4, 0),
                       span=10, precision=1e-8, verbosity=1000, steps=5000)


def storeTiny3():
    name, m, n, p = 'tiny3', 1, 8, [3, 3]
    t = np.array([[3, 4, 5, 6, 7, 8, 9, 10], [4, 6, 8, 9, 10, 12, 14, 15], [4, 6, 8, 10, 11, 12, 13, 15]])
    x0 = np.array([[3, 5, 6, 8, 8, 9, 10, 12], [3, 5, 8, 4, 9, 14, 12, 16], [4, 5, 7, 9, 10, 9, 10, 12]])
    w = random_w(m, p)
    storeTestdata(name, m, n, p, t, w, x0)
    print(name, ' stored')


def tiny3():
    return Testmanager('tiny3',
                       ['identity'], ['didentity'],
                       eta=nnw_numpy.etaiterator(5e-4, 10000),
                       span=20, precision=1e-8, verbosity=1000, steps=50000)


def storeAll(s):
    for t in [storeTiny0, storeTiny1, storeTiny2, storeTiny3][s]:
        t()

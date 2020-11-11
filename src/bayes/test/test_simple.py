import unittest
from math import isclose

import numpy as np

import by.archive.bayes as mb
import by.archive.pomescratch.bayes as pb

a_implies_b = [1, 0], [[1, 0], [0, 1]]
a_implies_xb = [1, 0], [[0, 1], [1, 0]]
triv = 2 * [0.5], 2 * [2 * [0.5]]
morse = [0.7, 0.3], [[0.99, 0.01], [0.02, 0.98]]  # dot, dash; dot|dot, dash|dot, dot|dash, dash|dash

alarm1 = [0.999, 0.001], [[0.9999, 0.0001], [0.0001, 0.9999]]
alarm2 = [0.999, 0.001], [[0.999, 0.001], [0.001, 0.999]]
alarm3 = [0.999, 0.001], [[0.995, 0.005], [0.005, 0.995]]

cabs = [0.85, 0.15], [[0.8, 0.2], [0.2, 0.8]]
cabsrev = [0.71, 0.29], [[0.9577, 0.0423], [0.5862, 0.4138]]

cancer = [0.9986, 0.0014], [[0.88, 0.12], [0.27, 0.73]]
smoker = [0.95, 0.05], [[0.8, 0.2], [0.01, 0.99]]


class TestBayes(unittest.TestCase):

    def t_simple(self, fun):
        for tc in [triv, morse, alarm1, alarm2, alarm3, cabs, cabsrev, cancer, smoker]:
            evd, bwd = fun(*fun(*tc))
            x = np.concatenate((evd, bwd), axis=None)
            y = np.concatenate(tc, axis=None)
            self.assertTrue(all([isclose(x[i], y[i], abs_tol=1e-6) for i in range(len(x))]))

    def test_simple(self):
        print()
        for fun in [mb.simple, pb.simple]:
            self.t_simple(fun)

    def test_pome(self):
        evd, bwd = pb.simple(*morse)
        print(evd, bwd)

# Johannes Siedersleben, QAware GmbH, Munich, Germany
# 03/06/2020

import unittest

import numpy as np
from sklearn.covariance import empirical_covariance



class TestBasics(unittest.TestCase):
    def testCovariance(self):
        X = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                      [0, 0, 99], [0, 0, 0], [98, 0, 0]])

        Z = X - X.sum(axis=0) / X.shape[0]
        cov1 = Z.T.dot(Z) / X.shape[0]

        cov2 = empirical_covariance(X)

        self.assertTrue((np.allclose(cov1, cov2)))
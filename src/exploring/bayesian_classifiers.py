# Johannes Siedersleben, QAware GmbH, Munich, Germany
# 27/05/2020

# Exploring Bayesian Classifiers with scikit

import unittest
from sklearn import naive_bayes as nb
import numpy as np


class Testautograd(unittest.TestCase):
    def testBernoulli(self):
        bernoulli = nb.BernoulliNB()
        # print(bernoulli.get_params())

        n_classes = 8
        n_samples = 10
        n_features = 2

        rng = np.random.RandomState(1)
        X = rng.randint(8, size=(10, 2))
        y = np.arange(1, 11)
        bernoulli.fit(X, y)

        prediction = bernoulli.predict(X)
        # score = bernoulli.score(X[2:3], y)

        print(prediction)
        # print(score)

    def testMultinomial(self):
        multinomial = nb.MultinomialNB()
        print(multinomial.get_params())

        rng = np.random.RandomState(1)
        X = rng.randint(8, size=(10, 2))
        y = np.arange(1, 11)
        multinomial.fit(X, y)

        prediction = multinomial.predict(X)
        # score = bernoulli.score(X[2:3], y)

        print(prediction)
        # print(score)


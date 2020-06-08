import unittest
import numpy as np
from sklearn import naive_bayes as nb


def to_one_hot(y):
    """
    @param y:
    @return:
    """
    Y = np.zeros((len(y), max(y) + 1), dtype=int)
    Y[range(len(y)), y] = 1
    return Y


def from_one_hot(Y: np.array) -> np.array:
    """
    @param Y: a np.array of 0 and 1, rowsum(Y) = ones, Y.shape = (m, K)
    @return: np.array with shape (m,)
,   """
    return np.argmax(Y, axis=1)


class CatEncoder(object):
    def __init__(self, y):
        self.y = y
        categories = list(enumerate(set(y)))
        self.decoding = dict(categories)
        self.encoding = dict([[c, k] for k, c in categories])

    def encode(self, y=None):
        """
        @param y:
        @return:
        """
        if y is None:
            y = self.y
        return [self.encoding[c] for c in y]

    def decode(self, z):
        """
        @param z:
        @return:
        """
        return [self.decoding[k] for k in z]


def colsum(X):
    return X.sum(axis=0)[None, :]

def rowsum(X):
    return X.sum(axis=1)[:, None]


class MultinomialJS(object):
    def __init__(self, alpha=1.):
        self.alpha = alpha
        self.log_evidence = None
        self.log_prior = None
        self.log_likelihood = None

    def fit(self, X, y):
        Y = to_one_hot(y)
        n = X.shape[1]

        self.log_evidence = np.log(colsum(X) / X.sum())
        self.log_prior = np.log(colsum(Y) / Y.sum())
        C = Y.T.dot(X)
        self.log_likelihood = np.log(C + self.alpha / (rowsum(C) + self.alpha * n))

    def log_proba(self, X):
        return self.log_prior + X.dot(self.log_likelihood.T)

    def proba(self, X):
        return np.exp(self.log_proba(X))

    def predict(self, X):
        return np.argmax(self.log_proba(X), axis=1)


class TestMultinomial(unittest.TestCase):
    def test_one_hot(self):
        y = np.array((0, 1, 2, 2, 4, 4, 4))
        Y = to_one_hot(y)
        z = from_one_hot(Y)
        self.assertTrue(np.array_equal(y, z))

    def testEncoding(self):
        Y = [['a', 'c', 'c', 'y', 'v', 'a', 'c'],
             [1, 3, 5, 9]]
        for y in Y:
            enc = CatEncoder(y)
            a = enc.encode()
            b = enc.decode(a)
            self.assertListEqual(b, y)

    def testMultinomial1(self):
        n_train_samples = 6
        n_test_samples = 2
        n_features = 3
        n_categories = 4
        max_x = 3
        alpha = 1.

        # preliminaries
        ones = np.ones(n_train_samples, dtype=float)
        f = lambda x: sum(x) % n_categories
        rng = np.random.RandomState(1)

        # setting up X and y
        X = rng.randint(max_x + 1, size=(n_train_samples, n_features))
        y = [f(x) for x in X]  # y.shape == (n_train_samples,)
        enc = CatEncoder(y)

        for classifier in [MultinomialJS(alpha=alpha), nb.MultinomialNB(alpha=alpha)]:
            classifier.fit(X, enc.encode())
            prediction = classifier.predict(X)
            print()
            print(y)
            print(enc.decode(prediction))



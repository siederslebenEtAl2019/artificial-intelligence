import unittest

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def run_iris():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    print()
    print(clf.predict(X[:2, :]))
    print(clf.predict_proba(X[:2, :]))
    print(clf.score(X, y))


class MyTestCase(unittest.TestCase):
    def test_iris(self):
        run_iris()

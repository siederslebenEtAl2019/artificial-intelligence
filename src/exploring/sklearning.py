# Exploring sklearn
# Johannes Siedersleben, QAware GmbH, Munich
# 30/05/2020


import unittest
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn import datasets

import seaborn as sns
import scipy

class TestSklearn(unittest.TestCase):
    def test0(self):
        iris = datasets.load_iris()
        digits = datasets.load_digits()

    def test1(self):
        x = np.arange(1, 6)
        y = np.array([4, 2, 1, 3, 7])

        X1 = x[:, np.newaxis]
        model1 = LinearRegression().fit(X1, y)
        yfit1 = model1.predict(X1)

        poly = PolynomialFeatures(degree=3, include_bias=False)
        X2 = poly.fit_transform(X1)
        model2 = LinearRegression().fit(X2, y)
        yfit2 = model2.predict(X2)

        plt.scatter(x, y)
        plt.show()
        plt.plot(x, yfit1)
        plt.show()
        plt.plot(x, yfit2)
        plt.show()

    def test2(self):
        data = fetch_20newsgroups()
        categories = ['talk.religion.misc', 'soc.religion.christian',
                      'sci.space', 'comp.graphics']
        train = fetch_20newsgroups(subset='train', categories=categories)
        test = fetch_20newsgroups(subset='test', categories=categories)

        # print(train.data[5])

        model = make_pipeline(TfidfVectorizer(), MultinomialNB())

        model.fit(train.data, train.target)
        labels = model.predict(test.data)
        cm = confusion_matrix(test.target, labels)
        sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False,
                    xticklabels=train.target_names, yticklabels=train.target_names)
        plt.xlabel = ('true label')
        plt.ylabel = ('predicted label')
        plt.show()

    def test3(self):
        n_train_samples = 100000
        n_test_samples = 10
        n_features = 9
        n_categories = 5
        max_x = 1

        f = lambda x: sum(x) % n_categories
        classifierB = BernoulliNB()
        classifierM = MultinomialNB(alpha=1.e-5)
        classifierG = GaussianNB()

        classifier = classifierM
        print(classifier.get_params())

        rng = np.random.RandomState(1)

        X = rng.randint(max_x + 1, size=(n_train_samples, n_features))
        y = [f(x) for x in X]
        classifier.fit(X, y)
        score = classifier.score(X, y)
        print('\n', score, '\n')

        ones = np.ones(n_test_samples, dtype=float)
        Xtest = rng.randint(max_x + 1, size=(n_test_samples, n_features))
        ytest = [f(x) for x in Xtest]
        score = classifier.score(Xtest, ytest)
        prediction = classifier.predict(Xtest)
        proba = classifier.predict_proba(Xtest)
        check = np.sum(proba, axis=1)
        yy = np.argmax(proba, axis=1)

        self.assertTrue(np.allclose(ones, check))
        self.assertTrue(np.array_equal(yy, prediction))
        print('\n', score, '\n', proba, '\n', prediction - ytest)

    def test4(self):
        n_train_samples = 6
        n_test_samples = 2
        n_features = 3
        n_categories = 4
        max_x = 3
        alpha = 0.

        # preliminaries
        ones = np.ones(n_train_samples, dtype=float)
        f = lambda x: sum(x) % n_categories
        rng = np.random.RandomState(1)

        # setting up X and y
        X = rng.randint(max_x + 1, size=(n_train_samples, n_features))
        y = [f(x) for x in X]       # y.shape == (n_train_samples,)
        categories = list(set(y))   # eliminate categories which do not occur
        n_categories = len(categories)
        categories.sort()
        d = {categories[k]: k for k in range(len(categories))}
        y = np.array([d[k] for k in y])   # y.shape == (n_train_samples,)

        # testing the multinomial classifier
        # classifier = MultinomialNB(alpha=1.)
        # classifier.fit(X, y)
        # proba = classifier.predict_proba(X)
        # log_proba = classifier.predict_log_proba(X)
        # prediction = classifier.predict(X)
        # check = np.sum(proba, axis=1)
        # yy = np.argmax(log_proba, axis=1)
        # self.assertTrue(np.allclose(ones, check))
        # self.assertTrue(np.array_equal(yy, prediction))

        # testing my own classifier
        prior = np.bincount(y) / n_train_samples
        log_prior = np.log(prior)   # prior.shape == (n_categories,)

        evidence = X.sum(axis=0) / X.sum()
        log_evidence = np.log(evidence)  # evidence.shape == (n_features,)

        basic_proba = np.empty((n_categories, n_features), dtype=float)
        for k in range(n_categories):
            samples = X[y == k, :]
            basic_proba[k] = (samples.sum(axis=0) + alpha) / \
                             (samples.sum() + alpha * n_features)



        log_basic_proba = np.log(basic_proba)
        log_proba1 = log_prior + X.dot(log_basic_proba.T)
        prediction1 = np.argmax(log_proba1, axis=1)
        # self.assertTrue(np.array_equal(prediction1, prediction))

        # testing multinomial and my own classifier against test data
        X = rng.randint(max_x + 1, size=(n_test_samples, n_features))
        y = [1, 2]
        # log_proba = classifier.predict_log_proba(X)
        # prediction = classifier.predict(X)

        log_proba1 = log_prior + X.dot(log_basic_proba.T)
        prediction1 = np.argmax(log_proba1, axis=1)
        # self.assertTrue(np.array_equal(prediction1, prediction))
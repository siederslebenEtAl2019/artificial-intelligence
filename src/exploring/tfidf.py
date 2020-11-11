# Exploring tdifd with sklearn
# Johannes Siedersleben, QAware GmbH, Munich
# 10/06/2020


import unittest

import numpy as np
import numpy.linalg as LA
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


def tfidf(tf):
    df = np.count_nonzero(tf, axis=0)
    n = tf.shape[0]
    idf = 1 + np.log((1 + n) / (1 + df))
    result = tf * idf
    norm = LA.norm(result, axis=1)
    return result / norm[:, None]


class TestSklearn(unittest.TestCase):
    def testA(self):
        # X[i, j] = frequency of term j in doc i
        tf = [[3, 0, 1],
              [2, 0, 0],
              [3, 0, 0],
              [4, 0, 0],
              [3, 2, 0],
              [3, 0, 2]]

        XB = tfidf(np.array(tf))
        transformer = TfidfTransformer(smooth_idf=True)
        XC = transformer.fit_transform(tf)

        print(XB)
        print()
        print(XC.toarray())


    def test0(self):
        corpus = ['xx xx xx yy zz', 'xx']
        feature_names = ['xx', 'yy', 'zz']
        cvectorizer = CountVectorizer()
        X = cvectorizer.fit_transform(corpus)
        XA = X.toarray()
        print(XA)

        # X = (i, j) xij = count(wordj in doci)
        # X.shape = (count(docs), count(words considered))
        self.assertListEqual(cvectorizer.get_feature_names(), feature_names)
        X1 = cvectorizer.transform([' '.join(feature_names)])  # (0, 0) 1, (0, 1) 1, (0, 2) 1
        self.assertListEqual(list(X1.toarray()[0]), [1, 1, 1])
        self.assertEqual(cvectorizer.vocabulary_.get('xx'), 0)

        tf = XA / np.max(XA, axis=0)
        df = np.count_nonzero(XA, axis=0)
        idf = np.log(1 + XA.shape[0] / df)
        XB = tfidf(XA)
        print()
        print(tfidf)

        vectorizer = TfidfVectorizer(binary=True)
        X = vectorizer.fit_transform(corpus)
        print()
        print(X.toarray())

    def test1(self):
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        twenty_train = fetch_20newsgroups(subset='trainxx', categories=categories,
                                          shuffle=True, random_state=42)

        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform((twenty_train.data))
        transformer = TfidfTransformer(use_idf=False)
        transformer.fit(X_train)
        X_train_tf = transformer.transform(X_train)

    def test2(self):
        corpus = [
            'xx xx xx yy',
            'xx zz zz zz'
        ]
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(corpus)
        print(X)
        # X[i, j] : token j appears in in document i that many times
        self.assertEqual(X[0, 0], 3)
        self.assertEqual(X[1, 0], 1)
        self.assertEqual(X[0, 2], 0)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        print(X)

    def test3(self):
        vectorizer = CountVectorizer()
        analyze = vectorizer.build_analyzer()
        tokens = analyze('xx xx xx yy zz q ; :')
        self.assertListEqual(['xx', 'xx', 'xx', 'yy', 'zz'], tokens)

        corpus = [
            'xx xx xx yy',
            'xx zz zz zz'
        ]
        vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names() # 'xx', 'yy', 'zz'
        self.assertListEqual(['xx', 'yy', 'zz'], feature_names)

        token_xx = vectorizer.vocabulary_.get('xx')
        token_yy = vectorizer.vocabulary_.get('yy')
        token_zz = vectorizer.vocabulary_.get('zz')
        token_uk = vectorizer.vocabulary_.get('uk')
        self.assertEqual(0, token_xx)
        self.assertEqual(1, token_yy)
        self.assertEqual(2, token_zz)
        self.assertEqual(None, token_uk)

    def test4(self):
        vectorizer = CountVectorizer(ngram_range=(1,2))
        analyze = vectorizer.build_analyzer()
        tokens = analyze('xx xx xx yy zz q ; :')
        expected = ['xx', 'xx', 'xx', 'yy', 'zz', 'xx xx', 'xx xx', 'xx yy', 'yy zz']
        self.assertListEqual(expected, tokens)

        corpus = [
            'xx xx xx yy',
            'xx zz zz zz'
        ]
        vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names()
        expected = ['xx', 'xx xx', 'xx yy', 'xx zz', 'yy', 'zz', 'zz zz']
        self.assertListEqual(expected, feature_names)

        token_xx_xx = vectorizer.vocabulary_.get('xx xx')
        self.assertEqual(1, token_xx_xx)

    def test5(self):
        x = [[1, 0, 5],
             [0, 2, 1]]
        x = np.array(x)
        nx = LA.norm(x, axis=1)
        x = x / nx[:, None]

        nx = LA.norm(x, axis=1)
        self.assertAlmostEqual(2., nx.sum())

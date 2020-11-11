import unittest

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def conf(x, y):
    K = max(x) + 1
    cm = np.zeros((K, K), dtype=int)
    for i in range(len(x)):
       cm[x[i], y[i]] += 1
    return cm


def makeConfusion(x, y, names):
    cm = confusion_matrix(x, y, normalize=None)
    vmax = cm.max()
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=True,
                xticklabels=names, yticklabels=names, vmin=0, vmax=vmax, cmap="YlGnBu")
    plt.xlabel = ('true label')
    plt.ylabel = ('predicted label')
    plt.show()


class MyTestCase(unittest.TestCase):
    def testConfusion(self):
        x = [0, 1, 2, 1, 2, 1, 0, 2, 1]
        y = [0, 1, 2, 1, 2, 1, 0, 2, 2]
        names = ['a', 'b', 'c']
        makeConfusion(x, y, names)

        x = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
        y = [1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1]
        names = ['negative', 'positive']
        makeConfusion(x, y, names)



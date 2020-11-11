# Johannes Siedersleben, QAware GmbH, Munich
# 14.12.2019
#
# This file contains utilities for dealing with digraphs

import operator
import random
import sys

import numpy as np

from bayes.factory import factory
from bayes.util.binaries import log2, int2bin

epsilon = 1e-10


def backprop(bst, k):
    memory = {k: ''}

    def process(j):
        if j in memory:
            return memory[j]

        result = ''
        for i in bst[j]:
            if i in memory:
                tail = memory[i]
            else:
                tail = process(i)
                memory[i] = tail
            result += 'dx' + str(j) + '/dx' + str(i) + ' * ' + tail
        return result

    return process(k)




def forward_star(bst):
    """
    :param bst: a backward star. bst is a list of lists, bst[i] contains all predecessors of i
    :return: the corresponding forward star fst. fst[i] contains all successors of i
    """
    n = len(bst)
    fst = []
    for j in range(n):
        fst.append([])
    for j in range(n):
        for i in bst[j]:
            fst[i].append(j)
    return [tuple(s) for s in fst]


def matches(bst, fpt):
    """
    :param bst: a backward star
    :param fpt: a forward probability table
    :return: true if bst matches fpt
    """
    if len(bst) != len(fpt):
        return False
    for i in range(len(bst)):
        if len(fpt[i]) != 2 ** len(bst[i]):
            return False
    return True


def mirror(bst):
    n = len(bst)
    if n == 0:
        return []
    fst = forward_star(bst)
    fst.reverse()
    for i, preds in enumerate(fst):
        aux = [n - j - 1 for j in preds]
        aux.sort()
        fst[i] = aux
    return fst


def bst_equals(bst1, bst2):
    """
    compare bst1 with bst2
    :param bst1: a backward star
    :param bst2: another backward star
    :return: True if they represent the same digraph,
    that is: bst1 equals bst2 elementwise
    """
    if len(bst1) != len(bst2):
        return False
    for i in range(len(bst1)):
        if len(bst1[i]) != len(bst2[i]):
            return False
        for j in range(len(bst1[i])):
            if bst1[i][j] != bst2[i][j]:
                return False
    return True


def bst_nbr_of_edges(bst):
    """
    :param bst: a backward star
    :return: number of edges of corresponding graph
    """
    return sum((len(bst[i]) for i in range(len(bst))))


def is_symmetric(bst):
    """
    :param bst: a backward star
    :return: true if bst equals its mirror
    """
    return bst_equals(bst, mirror(bst))


def sources(bst):
    """
    :param bst: backward star
    :return: tuple of sources (nodes with no predecessors)
    """
    return (i for i in range(len(bst)) if len(bst[i]) == 0)


def sinks(bst):
    """
    :param bst: backward star
    :return: tuple of sources (nodes with no predecessors)
    """
    return sources(forward_star(bst))


def count_cut(bst, m):
    """
    :param bst: a forward star of topologically ordered digraph
    :param k: defines a cut (S, T): S = range(k), T = range(k, n)
    :return: number of edges from S to T
    """
    return sum((len([i for i in bst[j] if i < m]) for j in range(m, len(bst))))


def min_cut(bst):
    """
    :param bst: a backward star of topologically ordered digraph
    :return: a cut k minimizing the size of the associated jpt
    """
    n = len(bst)

    def weight(m, k):
        return 2 ** m + 2 ** (k + n - m)

    ws = (weight(m, count_cut(bst, m)) for m in range(n))
    min_index, min_value = min(enumerate(ws), key=operator.itemgetter(1))
    return min_index, min_value


def min_cut2(bst):
    n = len(bst)

    def weight(m0, m1, k, l):
        return 2 ** m0 + 2 ** (k + m1 - m0) + 2 ** (l + n - m1)

    ws = [[m0, m1, weight(m0, m1, count_cut(bst, m0), count_cut(bst, m1))] for m0 in range(n) for m1 in range(m0, n)]
    min_idx, min_value = min(enumerate((x[2] for x in ws)), key=operator.itemgetter(1))
    return ws[min_idx][0], ws[min_idx][1], min_value


def cut(bst, k):
    """
    :param bst: a topologically ordered backward star
    :param k: first node of right half
    :return: left half (bst1), link, right half (bst2)
    link is a dictionary:
    keys: nodes on the right with predecessors on the left
    values: list of predecessors on the left.
    """
    n = len(bst)
    if k >= n:
        raise ValueError
    bst0 = bst[:k]
    bst1 = bst[k:]
    link = {}
    for j in range(k, n):
        aux = [i for i in bst[j] if i < k]
        if len(aux) > 0:
            link[j - k] = aux
    return bst0, link, bst1


def union(bst1, link, bst2):
    """
    :param bst1: a backward  star
    :param link: links bst1 and bst2
    :param bst2: a backwardstar
    :return: union of bst1, link and bst2
    """
    bst = []
    k = len(bst1)
    for i in range(len(bst1)):
        bst.append(bst1[i])
    for i in range(len(bst2)):
        bst.append(bst2[i])
    for i, pred in link:
        bst[i + k].append(pred)
    return bst


def gen_bst_layers(length, width):
    if length < 2:
        raise ValueError
    n = (length - 2) * width + 2
    bst = [None for i in range(n)]
    columns = [None for i in range(length)]
    columns[0] = (0,)
    bst[0] = ()
    for j in range(1, length - 1):
        columns[j] = range((j - 1) * width + 1, j * width + 1)
        for i in columns[j]:
            bst[i] = tuple(columns[j - 1])
    bst[n - 1] = tuple(columns[length - 2])
    return tuple(bst)


def gen_bst_full(n):
    return tuple((tuple(range(j)) for j in range(n)))


def gen_fpt(bst):
    return [[random.uniform(0, 1) for i in range(2 ** len(bst[j]))] for j in range(len(bst))]


def gen_layers(type, length, width):
    bst = gen_bst_layers(length, width)
    fpt = gen_fpt(bst)
    return factory.make(type, 'layer' + str(length) + str(width), bst, fpt)


def gen_full(type, n):
    bst = gen_bst_full(n)
    fpt = gen_fpt(bst)
    return factory.make(type, 'full' + str(n), bst, fpt)


def fpt2np(fpt):
    """
    :param fpt: a linear forward probability table
    :return: a shaped forward probability table
    """
    n = len(fpt)
    result = [None] * n
    for i in range(n):
        slen = len(fpt[i])
        size = log2(slen)
        shape = (1,) if size == 0 else (2,) * size
        result[i] = np.zeros(shape)
        for s in range(slen):
            b = int2bin(s, size)
            result[i][tuple(b)] = fpt[i][s]
    return result


def fpt_diff(fpt1, fpt2):
    diff = 0
    if len(fpt1) != len(fpt2):
        return sys.maxsize
    for i in range(len(fpt1)):
        if len(fpt1[i]) != len(fpt2[i]):
            return sys.maxsize
        for j in range(len(fpt1[i])):
            diff = max(abs(fpt1[i][j] - fpt2[i][j]), diff)
    return diff


def fpt_equals(fpt1, fpt2):
    """
    compare fpt1 with ftp2
    :param fpt1: a forward probability table
    :param fpt2: another forward probability table
    :return: if they agree in shape and values up to epsilon
    """
    return fpt_diff(fpt1, fpt2) <= epsilon

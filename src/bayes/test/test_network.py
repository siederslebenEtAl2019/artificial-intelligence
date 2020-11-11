import unittest

import numpy as np

import bayes.factory.factory as factory
from bayes.util.binaries import *
from bayes.util.digraphs import gen_full, gen_layers, mirror

epsilon = 1e-12
type = 'bnp'

bst_empty = ()
empty = 'empty', bst_empty, []

bst_single = ((),)
single = 'single', bst_single, [[0.1]]

bst_simple = [[], [0]]
simple = 'simple', bst_simple, \
         ((0.7,), (0.2, 0.4))  # forward probability table. fpt[i] = table of node i; contains 2^k entries
a_implies_b = 'a implies b', bst_simple, [[1], [1, 0]]
a_implies_xb = 'a implies xb', bst_simple, [[1], [0, 1]]
triv = 'triv', bst_simple, [[0.5], [0.5, 0.5]]
morse = 'morse', bst_simple, [[0.7], [0.99, 0.02]]
alarm1 = 'alarm1', bst_simple, [[0.999], [0.9999, 0.0001]]
alarm2 = 'alarm2', bst_simple, [[0.999], [0.999, 0.001]]
alarm3 = 'alarm3', bst_simple, [[0.999], [0.995, 0.005]]
cabs = 'cabs', bst_simple, [[0.85], [0.8, 0.2]]
revcabs = 'revcabs', bst_simple, [[0.71], [0.9577, 0.5862]]
cancer = 'cancer', bst_simple, [[0.9986], [0.88, 0.27]]
smoker = 'smoker', bst_simple, [[0.95], [0.8, 0.01]]

bst_fork = ((), (0,), (0,))
fpt_fork = [[0.7], [0.1, 0.6], [0.8, 0.3]]
fork = 'fork', bst_fork, fpt_fork

bst_collider = ((), (), (0, 1))
fpt_collider = [[0.7], [0.4], [0.1, 0.6, 0.8, 0.3]]
collider = 'collider', bst_collider, fpt_collider

bst_chain1 = ((), (0,), (1,))
fpt_chain1 = ([0.1], [0.2, 0.6], [0.3, 0.9])
chain1 = 'chain1', bst_chain1, fpt_chain1

bst_chain2 = ((), (0,), (1,), (2,))
fpt_chain2 = ([0.1], [0.2, 0.6], [0.3, 0.9], [0.1, 0.8])
chain2 = 'chain2', bst_chain2, fpt_chain2

bst_triangle = ((), (0,), (0, 1))
fpt_triangle = [[0.3], [0.1, 0.6], [0.2, 0.5, 0.1, 0.8]]
triangle = 'triangle', bst_triangle, fpt_triangle

bst_diamond1 = ((), (0,), (0,), (1, 2))
fpt_diamond1 = [[0.7], [0.1, 0.8], [0.8, 0.1], [0.05, 0.5, 0.5, 0.95]]
diamond1 = 'diamond1', bst_diamond1, fpt_diamond1

bst_diamond2 = ((), (0,), (0, 1), (1, 2))
fpt_diamond2 = [[0.7], [0.1, 0.8], [0.4, 0.2, 0.2, 0.7], [0.05, 0.5, 0.5, 0.95]]
diamond2 = 'diamond2', bst_diamond2, fpt_diamond2

bst_redundant2 = ((), (0,), (0,), (1, 2))
fpt_redundant2 = [[0.7], [0.99, 0.01], [0.99, 0.01], [0.99, 0.5, 0.5, 0.1]]
redundant2 = 'redundant2', bst_redundant2, fpt_redundant2

bst_redundant3 = ((), (0,), (0,), (0,), (1, 2, 3))
fpt_redundant3 = [[0.5], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01],
                  [0.999, 0.999, 0.999, 0.001, 0.999, 0.001, 0.001, 0.001]]
redundant3 = 'redundant3', bst_redundant3, fpt_redundant3

# bst6 = ((), (0,), (0, 1), (1, 2), (1, 2, 3), (3, 4))
# fpt6 = ([0.7], [0.2, 0.4], [0.1, 0.3, 0.5, 0.8], [0.3, 0.4, 0.9, 0.1],
#         [0.3, 0.4, 0.9, 0.1, 0.3, 0.4, 0.9, 0.1], [0.3, 0.4, 0.9, 0.1])

bst6 = ((), (0,), (0, 1), (0, 1, 2), (1, 2, 3), (3, 4))
fpt6 = ([0.7], [0.2, 0.4], [0.1, 0.3, 0.5, 0.8], [0.7, 0.4, 0.1, 0.2, 0.3, 0.4, 0.9, 0.1],
        [0.3, 0.4, 0.9, 0.1, 0.3, 0.4, 0.9, 0.1], [0.3, 0.4, 0.9, 0.1])
six_nodes = 'six nodes', bst6, fpt6

bst11 = ((), (0,), (0,), (0,), (1, 2), (1, 2, 3), (2, 3), (4, 5), (4, 5, 6), (5, 6), (7, 8, 9))

testcases0 = [six_nodes]
testcases1 = [redundant2]
testcases2 = [triv, morse, alarm1, alarm2, alarm3, cabs, revcabs, cancer, smoker,
              fork, collider, chain1, chain2, triangle, diamond1, diamond2,
              redundant2, redundant3,
              six_nodes]


def print_jpt(network):
    print('\njpt of ' + network.name)
    for idx in binary_cube([None] * len(network)):
        print(bin2int(idx), '\t', idx, '\t', network.jpt[tuple(idx)])


def print_detail(network):
    print('\nDetails of ', network.name)
    print('bst:\t', network.bst)
    print('mirror(bst):\t', mirror(network.bst))
    print('fst:\t', network.forward_star())
    print('fpt:\t', network.forward_probability_table())
    bpt = network.backward_probability_table(full=False)
    bptf = network.backward_probability_table(full=True)
    print('\nbpt:')
    for i, x in enumerate(bpt):
        print(i, x)
    print('\n\nbptf:')
    for i, x in enumerate(bptf):
        print(i, x)
    print_jpt(network)


class TestNetwork(unittest.TestCase):

    def test_probabilities(self):
        nw = factory.make(type, *morse)
        n = len(nw.backward_star())
        print('\n\n' + nw.name + '\n')

        print('joint_probabilities_internal')
        for values in ((), (0,), (1, 0), (1, 1)):
            print(values, nw.joint_probability_internal(values))

        print('joint_probabilities')
        for pattern in ((0, 0), (None, 0), (0, None), (None, None)):
            print(pattern, nw.joint_probability(pattern))

        print('conditional_probabilities')
        for pattern in ((0, 0), (None, 0), (0, None), (None, None)):
            for given in ((), (0,), (0, 1)):
                print(pattern, given, nw.conditional_probability(pattern, given))

    def test_basics(self):
        for tc in testcases0:
            nw = factory.make(type, *tc)
            self.t_network(nw)

    def test_basics_on_full(self):
        test_networks = []
        # for k in (1, 2, 3, 4, 16):
        for k in (17,):
            test_networks.append(gen_full(type, k))
        for nw in test_networks:
            self.t_network(nw)

    def test_basics_on_layers(self):
        test_networks = []
        # for i, j in ((2, 1), (2, 2), (2, 5), (4, 4), (8, 1), (8, 2)):
        # for i, j in ((8, 3), ):
        for i in range(2, 4):
            for j in range(1, 7):
                test_networks.append(gen_layers(type, i, j))
        for nw in test_networks:
            self.t_network(nw)

    def t_network(self, network):
        ff = network.mirror(full=True).mirror(full=False)
        gg = network.mirror(full=False).mirror(full=False)
        diff_full = np.amax(abs(ff.jpt - network.jpt))
        diff = np.amax(abs(gg.jpt - network.jpt))
        print('\n', network.name, network.is_symmetric())
        print('diff_full: ', diff_full)
        print('diff: ', diff)

    def test_mirror(self):
        for network in testcases0:
            w = factory.make(type, *network)
            m = w.mirror(full=False)
            f = w.mirror(full=True)
            mm = w.mirror(full=False).mirror(full=False)
            fm = w.mirror(full=False).mirror(full=True)
            mf = w.mirror(full=True).mirror(full=False)
            ff = w.mirror(full=True).mirror(full=True)
            fmm = w.mirror(full=True).mirror(full=False).mirror(full=False)

            print()
            for x in [w, m, f, mm, fm, mf, ff, fmm]:
                print(x)
                print_jpt(x)

            print('\n\n\n\n\n\n')

    def test_detail(self):
        for network in testcases0:
            w = factory.make(type, *network)
            print_detail(w)
            print('\n\n\n\n\n')

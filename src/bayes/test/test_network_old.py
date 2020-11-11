import unittest
from math import isclose

from bayes.util.binaries import fpt_diff
from bayes.util.digraphs import *
from search.conjoin import conjoin

epsilon = 1e-5

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

bst_chain = ((), (0,), (1,))
fpt_chain = ([0.7], [0.8, 0.1], [0.6, 0.4])
chain = 'chain', bst_chain, fpt_chain

bst_diamond = ((), (0,), (0,), (1, 2))
fpt_diamond = [[0.7], [0.1, 0.8], [0.8, 0.1], [0.05, 0.5, 0.5, 0.95]]
diamond = 'diamond', bst_diamond, fpt_diamond

bst_diamond2 = ((), (0,), (0, 1), (1, 2))
fpt_diamond2 = [[0.7], [0.1, 0.8], [0.4, 0.2, 0.2, 0.7], [0.05, 0.5, 0.5, 0.95]]
diamond2 = 'diamond2', bst_diamond2, fpt_diamond2

bst_redundant2 = ((), (0,), (0,), (1, 2))
fpt_redundant2 = [[0.7], [0.99, 0.01], [0.99, 0.01], [1, 0.5, 0.5, 0.1]]
redundant2 = 'redundant2', bst_redundant2, fpt_redundant2

bst_redundant3 = ((), (0,), (0,), (0,), (1, 2, 3))
fpt_redundant3 = [[0.5], [0.99, 0.01], [0.99, 0.01], [0.99, 0.01], [1, 1, 1, 0, 1, 0, 0, 0]]
redundant3 = 'redundant3', bst_redundant3, fpt_redundant3

bst6 = ((), (0,), (0, 1), (1, 2), (1, 2, 3), (3, 4))
fpt6 = ([0.7], [0.2, 0.4], [0.1, 0.3, 0.5, 0.8], [0.3, 0.4, 0.9, 0.1],
        [0.3, 0.4, 0.9, 0.1, 0.3, 0.4, 0.9, 0.1], [0.3, 0.4, 0.9, 0.1])
six_nodes = 'six nodes', bst6, fpt6

bst11 = ((), (0,), (0,), (0,), (1, 2), (1, 2, 3), (2, 3), (4, 5), (4, 5, 6), (5, 6), (7, 8, 9))

testcases0 = [diamond2]
testcases1 = [redundant2]
testcases2 = [triv, morse, alarm1, alarm2, alarm3, cabs, revcabs, cancer, smoker,
              fork, collider, chain, redundant2, redundant3, six_nodes]


class TestNetwork(unittest.TestCase):

    def test_all(self):
        for tc in testcases0:
            network = factory.make('bnp', *tc)
            self.t_str(network)
            self.t_backward_probability_table(network)
            # self.t_joint_probability(network)
            # self.t_conditional_probability(network)
            # self.t_source_probability(network)
            # self.t_mirror(network)

    def test_all_gen(self):
        bst = gen_bst_layers(3, 4)
        fpt = gen_fpt(bst)
        network = factory.make('bnp', 'test', bst, fpt)
        self.t_str(network)
        self.t_joint_probability(network)
        self.t_conditional_probability(network)
        self.t_backward_probability_table(network)
        self.t_source_probability(network)
        self.t_mirror(network)

    @staticmethod
    def t_joint_probability_(network):
        print('\nTesting joint probabilities on ' + network.get_name())
        vs = [lambda: (0, 1)] * len(network)
        cum = 0
        for v in conjoin(vs):
            prob = network.joint_probability(v)
            cum += prob
        assert (isclose(cum, 1, abs_tol=1e-5))

    @staticmethod
    def t_joint_probability(network):
        print('\nTesting joint probability on ' + network.get_name())
        ps = [lambda: (0, 1, None)] * len(network)
        for p in conjoin(ps):
            print(f'{p}  {network.joint_probability(p):.6f}')

    @staticmethod
    def t_conditional_probability(network):
        print('\nTesting conditional probabilities on ' + network.get_name())
        ps = [lambda: (0, 1, None)] * len(network) + [lambda: range(len(network))]
        for p in conjoin(ps):
            print(p)
            cp = network.conditional_probability(p[:-1], (p[-1],))
            print(cp)
            print(f'{p}  {network.conditional_probability(p[:-1], (p[-1],)):.6f}')

    @staticmethod
    def t_backward_probability_table(network):
        print('\nTesting backward probability table on ' + network.get_name())
        result = network.backward_probability_table_full()
        for p in result:
            print(p)

    @staticmethod
    def t_source_probability(network):
        print('\nTesting source probability on ' + network.get_name())
        n1 = len(network.get_sources())
        n2 = len(network.get_sinks())
        ps = [lambda: (0, 1)] * (n1 + n2)
        for p in conjoin(ps):
            prob = network.source_probability(p[:n1], p[n1:])
            print(p[:n1], p[n1:], prob)

    @staticmethod
    def t_mirror(network):
        print('\nTesting mirror o mirror on ' + network.get_name())
        m = network.mirror()
        mm = m.mirror()
        diff = abs(fpt_diff(network.forward_probability_table(), mm.forward_probability_table()))
        print(diff)
        # if network.is_symmetric():
        #     assert (diff < 1e-3)

    @staticmethod
    def t_full_mirror(network):
        print('\nTesting full-mirror o mirror on ' + network.get_name())
        fm = network.mirror_full()
        print(fm)

    @staticmethod
    def t_str(network):
        print('\nTesting str on ' + network.get_name())
        print(network)

    def test_jpt(self):
        for tc in testcases0:
            network = factory.make('npb', *tc)
            m = network.mirror()
            mm = m.mirror()

            jpt1 = network.build_jpt()
            jpt2 = m.build_jpt()
            jpt3 = mm.build_jpt()

            print(network)
            print(jpt1)
            print(network.backward_probability_table_full())
            print(m)
            print(jpt2)
            print(m.backward_probability_table_full())
            print(mm)
            print(jpt3)
            print(mm.backward_probability_table_full())

    def test_pome(self):
        network = factory.make('pome', *chain)
        nw = network.pome_network

        # p = nw.marginal()
        # print('\nmarginal\n', p)

        print('\npredict_probability\n')
        # for i in [0, 1, None]:
        #         #     for j in [0, 1, None]:
        #         #         for k in [0, 1, None]:
        #         #             p = nw.predict_proba([i, j, k])
        #         #             print(i, j, k, p)
        zz = 0
        for i in [0, 1]:
            for j in [0, 1]:
                for k in [0, 1]:
                    p = network.joint_probability_internal([i, j, k])
                    zz += p
                    print(i, j, k, p)
        # self.assertAlmostEqual(zz, 1.0, 6)

        # print('predict_proba\n')
        # p = nw.predict_proba()
        # print(p)

        # q = [nw.predict_proba([None, i])[0] for i in range(2)]
        # p = [[q[j].probability(k) for k in range(2)] for j in range(2)]
        # print(p)

    def test_cut(self):
        # bst = gen_bst_layers(20,5)
        bst = bst_redundant3
        print('\ncut0: ', 2.0 ** len(bst))
        min_idx, min_val = min_cut(bst)
        print('\ncut1: ', min_idx, float(min_val))
        min_idx0, min_idx1, min_val = min_cut2(bst)
        print('\ncut2: ', min_idx0, min_idx1, float(min_val))

    def test_gen(self):
        bst = gen_bst_layers(12, 2)
        fpt = gen_fpt(bst)
        network = factory.make('npb', 'test', bst, fpt)
        print(network)

from bayes.test.test_network import *
from bayes.util.binaries import binary_cube
from bayes.util.digraphs import *


class TestDigraphs(unittest.TestCase):

    @staticmethod
    def test_fpt2np():
        print('\nforward star')
        for tc in [single, morse, chain1, chain2, fork, collider, diamond1, diamond2, six_nodes]:
            # for tc in [morse]:
            tw = factory.make('bnp', *tc)
            fpt = tw.forward_probability_table()
            fptnp = fpt2np(fpt)
            print(tw.name, fpt, fptnp)

    @staticmethod
    def test_fst():
        print('\nforward star')
        print('single', list(forward_star(bst_single)))
        print('simple', list(forward_star(bst_simple)))
        print('chain1', list(forward_star(bst_chain1)))
        print('fork', list(forward_star(bst_fork)))
        print('collider', list(forward_star(bst_collider)))
        print('redundant2', list(forward_star(bst_redundant2)))
        print('redundant3', list(forward_star(bst_redundant3)))
        print('six nodes', list(forward_star(bst6)))

    @staticmethod
    def test_sources():
        print('\nsources')
        print('basic', list(sources(bst_simple)))
        print('collider', list(sources(bst_collider)))
        print('redundant3', list(sources(bst_redundant3)))
        print('six nodes', list(sources(bst6)))

    @staticmethod
    def test_sinks():
        print('\nsinks')
        print('basic', list(sinks(bst_simple)))
        print('fork', list(sinks(bst_fork)))
        print('redundant3', list(sinks(bst_redundant3)))
        print('six nodes', list(sinks(bst6)))

    @staticmethod
    def t_mirror():
        print('\nmirror')
        print('single', mirror(bst_single))
        print('basic', mirror(bst_simple))
        print('fork', mirror(bst_fork))
        print('collider', mirror(bst_collider))
        print('chain', mirror(bst_chain1))
        print('six nodes', mirror(bst6))

    @staticmethod
    def test_is_symmetric():
        print('\nis_symmetric')
        print('single', is_symmetric(bst_single))
        print('basic', is_symmetric(bst_simple))
        print('fork', is_symmetric(bst_fork))
        print('collider', is_symmetric(bst_collider))
        print('six nodes', is_symmetric(bst6))
        print('redundant2', is_symmetric(bst_redundant2))
        print('redundant3', is_symmetric(bst_redundant3))

    def test_mirror_bst(self):
        print('\nmirror o mirror')
        for b in [bst_single, bst_simple, bst_chain1, bst6, bst_redundant2, bst_redundant3]:
            m = mirror(mirror(b))
            self.assertTrue(bst_equals(m, b))

    @staticmethod
    def test_binary_cube():
        print('\nbinary cube')
        pattern = (1, None, 1)
        for p in binary_cube(pattern):
            print(p)

    @staticmethod
    def test_cut():
        print('\ntesting cut')
        b = gen_bst_layers(5, 2)
        print('\nb', b)
        b0, link, b1 = cut(b, 5)
        print('cut', b0, link, b1)

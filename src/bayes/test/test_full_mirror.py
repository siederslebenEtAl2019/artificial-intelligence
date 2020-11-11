from bayes.factory import factory
import unittest


class TestFork(unittest.TestCase):

    def test_fm(self):
        bf = factory.make('npb', *triv)
        fm = bf.mirror_full()
        print(fm)

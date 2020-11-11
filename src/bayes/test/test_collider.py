import unittest

import bayes.archive.bayes as mb


class TestCollider(unittest.TestCase):

    def test_collider(self):
        morse1 = [0.7, 0.3, 1, 0], [0.99, 0.01, 0.99, 0.01, 0.02, 0.98, 0.02, 0.98]  # independent of A2
        morse2 = [0, 1, 0.7, 0.3], [0.99, 0.01, 0.99, 0.01, 0.02, 0.98, 0.02, 0.98]  # independent of A1
        coll1 = [0.9, 0.1, 1, 0], [0.9, 0.1, 0.9, 0.1, 0.1, 0.9, 0.1, 0.9]
        coll2 = [0.7, 0.3, 0.6, 0.4], [0.9, 0.1, 0.8, 0.2, 0.6, 0.4, 0.1, 0.9]

        for tc in [morse1, morse2]:
            print(mb.collider(*tc))

    def test_chain(self):
        pass

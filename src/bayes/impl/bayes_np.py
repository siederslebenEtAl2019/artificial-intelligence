# This program implements bayesian networks given by
# backward star (bst): a tuple of tuples containing each node's predecessors
# forward probability table (fpt): a list of conditional probability tables, one for each node

import numpy as np

import bayes.impl.abstract as abstract
from bayes.util.binaries import binary_cube


def build_jpt(network):
    shape = (2,) * len(network)
    jpt = np.zeros(shape)
    for p in binary_cube([None] * len(network)):
        jpt[tuple(p)] = network.joint_probability_internal(p)
    return jpt


class BayesianNetwork(abstract.BayesianNetwork):
    def __init__(self, name, bst, fpt):
        super().__init__(name, bst, fpt)
        self.jpt = build_jpt(self)

    def make(self, name, bst, fpt):
        return BayesianNetwork(name, bst, fpt)

    def joint_probability(self, pattern):  # overrides super.joint_probability
        """
        :param pattern: fixed variables are 0 or 1; free variables are None
        :return: probability of given pattern of fixed variables
        """
        idx = (slice(None) if pattern[i] is None else pattern[i] for i in range(len(self)))
        return self.jpt[tuple(idx)].sum()

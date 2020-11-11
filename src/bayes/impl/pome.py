# This program implements bayesian networks given by
# backward star (bst): a tuple of tuples containing each node's predecessors
# forward probability table (fpt): a list of conditional probability tables, one for each node


import pomegranate as pome

import bayes.impl.abstract as abstract
from util.binaries import int2bin


def make_pome_network(name, bst, fpt):
    """
    :param name: The network's name
    :param bst: The network's backward star
    :param fpt: The network's forward probability table
    :return: The corresponding pomegranate BayesianNetwork
    """
    network = pome.BayesianNetwork(name)
    n = len(bst)
    nodes = [None] * n
    distributions = [None] * n

    for j in range(n):
        if len(bst[j]) == 0:  # j has no predecessors
            p = fpt[j][0]
            distributions[j] = pome.DiscreteDistribution({0: p, 1: 1 - p})
            nodes[j] = pome.Node(distributions[j])
            network.add_nodes(nodes[j])
        else:
            tbl = [[] for i in range(2 * len(fpt[j]))]
            for (i, p) in enumerate(fpt[j]):
                b = int2bin(i, len(bst[j]))
                tbl[2 * i] = b + [0, p]
                tbl[2 * i + 1] = b + [1, 1 - p]
            distributions[j] = pome.ConditionalProbabilityTable(tbl, [distributions[i] for i in bst[j]])
            nodes[j] = pome.Node(distributions[j])
            network.add_nodes(nodes[j])
            for i in bst[j]:
                network.add_edge(nodes[i], nodes[j])
    network.bake()
    return network


class PomeBayesianNetwork(abstract.BayesianNetwork):
    def __init__(self, name, bst, fpt):
        super().__init__(name, bst, fpt)
        self.pome_network = make_pome_network(name, bst, fpt)

    def __str__(self):
        return super().__str__() + '\n' + str(self.pome_network)

    def make(self, name, bst, fpt):
        return PomeBayesianNetwork(name, bst, fpt)

    def joint_probability_internal(self, values):
        """
        :param pattern: fixed variables are 0 or 1
        :return: p(Xi=xi, i = 0,.., n-1)
        """
        return self.pome_network.probability(values)

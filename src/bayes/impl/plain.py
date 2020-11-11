# This program implements bayesian networks given by
# backward star (bst): a tuple of tuples containing each node's predecessors
# forward probability table (fpt): a list of conditional probability tables, one for each node


import by.impl.abstract as abstract


class BayesianNetwork(abstract.BayesianNetwork):
    def __init__(self, name, bst, fpt):
        super().__init__(name, bst, fpt)

    def make(self, name, bst, fpt):
        return BayesianNetwork(name, bst, fpt)

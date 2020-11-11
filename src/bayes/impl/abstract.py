#
# Johannes Siedersleben, QAware GmbH, December 2019
#
# This program implements Bayesian networks.
# A Bayesian network is given by its topology and its probabilities:
# backward star (bst): a tuple of tuples containing each node's predecessors
# forward probability table (fpt): a list of conditional probability tables, one for each node
# The backward star is a list containing each node's predecessor. Examples are bst_simple, bst_chain
# and others.
# The forward probability table is organized as follows:
# Let i be a node with k predecessors (k >= 0). Then fpt[i] contains 2 ** k entries
# in this order
# p(Xi = 0 | PX0 = 0, ..., PXk-1 = 0)
# ...
# p(Xi = 0 | PX0 = 1, ..., PXk-1 = 1)
#
# So, with k = 3 (three predecessors) we get:
# fpt[i][0] = p(Xi = 0 | PX0 = 0, PX1 = 0, PX2 = 0)
# ...
# fpt[i][5] = p(Xi = 0 | PX0 = 1, PX1 = 0, PX2 = 1)
# ...
# fpt[i][7] = p(Xi = 0 | PX0 = 1, PX1 = 1, PX2 = 1)
#
# Node 0 never has predecessors, so there is only one entry
# fpt[0][0] = p(X0 = 0)
#
# Examples are fpt_simple, fpt_chain and others.
#

from by.util.binaries import *
from by.util.digraphs import *


class BayesianNetwork(object):
    def __init__(self, name, bst, fpt):
        if len(bst) == 0:
            raise ValueError
        if not matches(bst, fpt):
            raise ValueError

        self.bst = list(bst)  # self.backward_star()
        self.fst = forward_star(bst)  # self.backward_star()
        self.fpt = list(fpt)  # self.forward_probability_table()
        self.name = name
        self.sources = list(sources(bst))
        self.sinks = list(sinks(bst))

    def __len__(self):
        return len(self.bst)

    def get_name(self):
        return self.name

    def forward_star(self):
        return list(self.fst)

    def backward_star(self):
        return list(self.bst)

    def is_symmetric(self):
        return is_symmetric(self.bst)

    def forward_probability_table(self):
        return list(self.fpt)

    def get_sources(self):
        return list(self.sources)

    def get_sinks(self):
        return list(self.sinks)

    def __eq__(self, other):
        t1 = bst_equals(self.backward_star(), other.backward_star())
        t2 = fpt_equals(self.forward_probability_table(), other.forward_probability_table())
        return t1 and t2

    def __str__(self):
        return '\n' + self.get_name() + '\n' + str(self.backward_star()) + '\n' + str(self.forward_probability_table())

    def make(self, name, bst, fpt):
        raise NotImplementedError

    def type(self):
        raise NotImplementedError

    def joint_probability_internal(self, values):
        """
        :param values: binary values of X0, X1, ..., Xk; k < n = number of nodes
        :return: p(X0=x0, ..., Xn=xk)
        joint_probability_internal(x0, x1, .., xk-1) = joint_probability(x0, x1, .., xk-1, None, ..., None)
        """
        n = len(values)
        p = 1
        for k in range(n):
            b = (values[i] for i in self.bst[k])  # values of predecessors of k
            i = bin2int(b)  # get corresponding index in fpt[k]
            fp = self.forward_probability_table()[k][i]
            p *= fp if values[k] == 0 else 1 - fp
        return p

    def joint_probability(self, pattern):
        """
        :param pattern: fixed variables are 0 or 1; free variables are None
        :return: probability of given pattern of fixed variables
        """
        return sum(self.joint_probability_internal(values) for values in binary_cube(pattern))

    def conditional_probability(self, pattern, given):
        """
        :param pattern: fixed and conditioned variables are 0 or 1; free variables are None
        :param given: list of given variables, eg (2, 3)
        preconditions:
        len(pattern) == len(self)
        0 <= len(given) <= len(pattern)
        if Xi is given then pattern[i] is not None
        :return: p(Xi = xi for fixed i | Xj = xj for given j)
        This is Bayes' formula:
        p(Xi = xi for fixed i | Xj = xj for given j) =
             p(Xi = xi for fixed i and Xj = xj for given j) /  p(Xj = xj for given j)
        """
        given = list(given)
        numerator = self.joint_probability(pattern)
        if len(given) == 0:
            denominator = 1
        else:
            denominator_pattern = [None] * len(pattern)
            for i in given:
                denominator_pattern[i] = pattern[i]
            denominator = self.joint_probability(denominator_pattern)
        return numerator / denominator

    def source_probability(self, source_values, sink_values):
        """
        :param source_values:
        :param sink_values:
        :return:
        """
        pattern = [None] * len(self)
        v = iter(source_values)
        for i in self.get_sources():
            pattern[i] = next(v)
        w = iter(sink_values)
        for i in self.get_sinks():
            pattern[i] = next(w)
        return self.conditional_probability(pattern, self.get_sinks())

    def backward_probability_table(self, full):
        """
        :param full: if full is False, the successors are the ones given bx forward star (fst)
        if full is True, for node i all nodes from i+1 through n-1 are taken as successors
        :return: the backward probability table
        # This table is organized as follows:
        # Let i be a node with k successors (k >= 0). Then bpt[i] contains 2 ** k entries
        # in this order
        # p(Xi = 0 | SX0 = 0, ..., SXk-1 = 0)
        # ...
        # p(Xi = 0 | SX0 = 1, ..., SXk-1 = 1)
        #
        # So, with k = 3 (three successors) we get:
        # bpt[i][0] = p(Xi = 0 | PX0 = 0, PX1 = 0, PX2 = 0)
        # bpt[i][5] = p(Xi = 0 | PX0 = 1, PX1 = 0, PX2 = 1)
        # bpt[i][7] = p(Xi = 0 | PX0 = 1, PX1 = 1, PX2 = 1)
        #
        # Node n-1 never has predecessors, so there is only one entry
        # bpt[n-1][0] = p(Xn-1 = 0)
        """
        n = len(self)
        bpt = [[] for i in range(n)]  # the future bpt
        successors = [list(range(k + 1, n)) for k in range(n)] if full else self.fst
        for i in range(n):
            pattern = [None] * n  # pattern for conditional probability
            pattern[i] = 0
            slen = len(successors[i])
            if slen == 0:  # compute p(xi = 0) (unconditional probability)
                bpt[i].append(self.conditional_probability(pattern, successors[i]))
            else:  # enumerate values v of successors of i
                for v in binary_cube([None] * slen):
                    for t in range(slen):
                        pattern[successors[i][t]] = v[t]
                    bpt[i].append(self.conditional_probability(pattern, successors[i]))
        return bpt

    def mirror(self, full):
        """
        :param full:
        :return: a bayesian network. Its backward star is mirror(self.bst),
        its forward probability table is backward probability table(self, full)
        Full=True: mirror contains the full set of edges
        Full=False: mirror contains the set of reversed edges of self
        """
        if full:
            name = self.get_name() + '-mirror-full'
            bst = gen_bst_full(len(self.bst))
            fpt = self.backward_probability_table(full=True)
        else:
            name = self.get_name() + '-mirror'
            bst = mirror(self.bst)
            fpt = self.backward_probability_table(full=False)
        fpt.reverse()
        for i in range(len(bst)):
            slen = len(fpt[i])
            size = log2(slen)
            fpt[i] = [fpt[i][t] for t in [rev_int(s, size) for s in range(slen)]]
        return self.make(name, bst, fpt)

import bayes.impl.bayes_np as bnp
import bayes.impl.plain as simple
import bayes.impl.pome as pome


def make(kind, name, bst, fpt):
    if kind == 'bnp':
        return bnp.BayesianNetwork(name, bst, fpt)
    elif kind == 'pome':
        return pome.PomeBayesianNetwork(name, bst, fpt)
    elif kind == 'plain':
        return simple.BayesianNetwork(name, bst, fpt)

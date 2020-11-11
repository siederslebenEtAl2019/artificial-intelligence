import unittest


def backprop(bst, k):
    memory = {}

    def process(j):
        if j in memory:
            return memory[j]

        formula = ''
        for i in bst[j]:
            if i in memory:
                tail = memory[i]
            else:
                tail = process(i)
                tail = tail if len(tail) == 0 else ' * ' + tail
                memory[i] = tail
            formula += 'dx' + str(j) + '/dx' + str(i) + tail + ' + '
        formula = formula[:-3]
        formula = '(' + formula + ')' if len(bst[j]) > 1 else formula
        return formula

    grad = process(k)
    if len(grad) > 0 and grad[0] == '(':
        grad = grad[1:-1]
    return grad


def backpropLatex(bst, k):
    memory = {}

    def newtail(i, j, tail):
        return "\\frac{\partial x_" + str(j) + "}{\partial x_" + str(i) + "}" + tail + "+"

    def process(j):
        if j in memory:
            return memory[j]

        formula = ''
        for i in bst[j]:
            if i in memory:
                tail = memory[i]
            else:
                tail = process(i)
                tail = tail if len(tail) == 0 else '\cdot' + tail
                memory[i] = tail
            formula += newtail(i, j, tail)
        formula = formula[:-1]
        formula = formula if len(bst[j]) <= 1 else '(' + formula + ')'
        return formula

    grad = process(k)
    if len(grad) > 0 and grad[0] == '(':
        grad = grad[1:-1]
    grad = newtail(0, len(bst) - 1, '')[:-1] + '=' + grad
    return '$\\begin{align}\\begin{split}' + grad + '\end{split}\end{align}$'


class TestBackpropagation(unittest.TestCase):

    def testBackprop(self):
        bst = [None] * 7
        bst[0] = ((), (), (0, 1))
        bst[1] = ((), (0,))
        bst[2] = ((), (0,), (1,))
        bst[3] = ((), (0,), (1,), (2,))
        bst[4] = ((), (), (0, 1))
        bst[5] = ((), (0,), (0,), (1, 2))
        bst[6] = ((), (0,), (0, 1), (1, 2))
        bst[7] = ((), (0,), (0, 1), (1, 2), (2, 3), (3, 4), (5,))

        for b in bst[7:8]:
            # grad = backprop(b, len(b) - 1)
            grad = backpropLatex(b, len(b) - 1)
            print(grad)

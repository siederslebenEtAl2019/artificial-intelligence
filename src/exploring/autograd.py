# Johannes Siedersleben, QAware GmbH, Munich, Germany
# 18/05/2020

# Exploring autograd

import unittest
import torch


def printNode(x):
    print('\n', x.data, '\n', x.grad, '\n', x.grad_fn)


class Testautograd(unittest.TestCase):

    def check(self, f, x):
        factor = 1e-4
        places = 4
        r = torch.randn(x.shape)
        dx = factor * r / r.norm()
        dz = f(x + dx) - f(x)
        self.assertAlmostEqual(dz.item(), dx.dot(x.grad).item(), places)

    def testScalar(self):
        """
        scalar --> scalar --> scalar
        This shows:
        * gradients along different paths are added
        * each forward step allows one backward step.

        """
        x = torch.tensor(1., requires_grad=True)

        s = x ** 2
        t = x ** 3
        y = s * t  # x ** 5 == 1

        y.backward()
        if x.grad != 5:  # 5 = 5 * x ** 4
            raise Exception

        with torch.no_grad():
            x += 1
            x.grad.zero_()   # is mandatory

        s = x ** 2
        t = x ** 3
        y = s * t  # x ** 5

        y.backward()
        if x.grad != 80:  # 80 = 5 * x ** 4
            raise Exception

    def testScalarScalar(self):
        """
        scalar * scalar --> scalar
        """
        x = torch.tensor(1., requires_grad=True)
        y = torch.tensor(2., requires_grad=True)
        z = torch.tensor(4., requires_grad=True)
        s = x + y
        t = y + z
        v = s * t

        v.backward(x)
        self.assertEqual(x.grad, 6)  # 6 = y + z
        self.assertEqual(y.grad, 9)  # 9 = 2 * y + x + z
        self.assertEqual(z.grad, 3)  # 3 = x + y

    def testVectorScalar(self):
        """
        vector * scalar --> vector
        """
        def f(x):
            return 3 * x.dot(torch.tensor([1., 2., 3.]))

        x = torch.tensor([3., 7., 11.], requires_grad=True)
        z = f(x)
        z.backward()
        self.check(f, x)

    def testVectorVector(self):
        """
        vector * vector --> scalar
        """
        x = torch.tensor([3., 4.], requires_grad=True)
        w = torch.tensor([2., 3.], requires_grad=True)

        y = w.dot(x)
        y.backward()
        self.assertTrue(x.grad.equal(w))
        self.assertTrue(w.grad.equal(x))

    def testMatrixMatrix(self):
        """
        matrix * matrix --> scalar
        """
        x = torch.tensor([[3., 4., 5], [6., 7., 8.]], requires_grad=True)
        w = torch.tensor([[13., 14., 15], [16., 17., 18.]], requires_grad=True)

        def f(x):
            return w.t().mm(x)

        y = f(x)
        y.backward()
        self.check(f, x)

        print()
        print(x.grad)
        print(w.grad)
        # self.assertTrue(x.grad.equal(w))
        # self.assertTrue(w.grad.equal(x))

    def testMatrixVector(self):
        """
        matrix * vector --> vector
        """
        x = torch.tensor([3., 4.], requires_grad=True)  # x : m x 1
        w = torch.tensor([[2., -3., 4.], [40., 50., -60.]],
                         requires_grad=True)  # w : m x n

        y = w.t().mv(x)  # y : n x 1
        r = torch.randn(3)  # r : n x 1
        y.backward(r)
        xg1 = x.grad
        wg1 = w.grad

        self.assertTrue(x.grad.equal(w.mv(r)))  # x.grad = Wr : m x 1
        self.assertTrue(w.grad.equal(x.ger(r)))  # x.grad = x outer r

        x.grad.zero_()
        w.grad.zero_()

        y = w.t().mv(x)  # y : n x 1
        z = y.dot(r)
        z.backward()
        print(x.grad - xg1)
        print(w.grad - wg1)

    def test4(self):
        """
        matrix * vector --> vector
        """
        for i in range(2):
            x = torch.tensor([3., 4., 5.], requires_grad=True)  # x : n x 1
            w = torch.tensor([[2., -3., 4.], [40., 50., -60.]],
                             requires_grad=True)  # w : m x n

            y = w.mv(x)  # y : m x 1
            r = torch.zeros(2)  # r : m x 1
            r[i] = 1
            y.backward(r)
            self.assertTrue(x.grad.equal(w.t().mv(r)))  # x.grad = WT r : n x 1
            self.assertTrue(w.grad.equal(x.ger(r).t()))  # w.grad = (x outer r)T

        x = torch.tensor([3., 4., 5.], requires_grad=True)  # x : n x 1
        w = torch.tensor([[2., -3., 4.], [40., 50., -60.]],
                         requires_grad=True)  # w : m x n

        y = w.mv(x)  # y : m x 1
        r = torch.randn(2)  # r : m x 1
        y.backward(r)
        self.assertTrue(x.grad.equal(w.t().mv(r)))  # x.grad = WT r : n x 1
        self.assertTrue(w.grad.equal(x.ger(r).t()))  # w.grad = (x outer r)T

    def test5(self):
        """
        matrix * matrix --> matrix
        """
        for i in range(2):
            for j in range(2):
                x = torch.tensor([[7., 9.], [11., 13.], [15., 17.]],
                                 requires_grad=True)  # n x k
                w = torch.tensor([[2., -3., 4.], [40., 50., -60.]],
                                 requires_grad=True)  # m x n
                y = w.mm(x)

                r = torch.zeros(2, 2)
                r[i, j] = 1
                y.backward(r)
                self.assertTrue(x.grad.equal(w.t().mm(r)))  # X.grad = WT r : n x k
                self.assertTrue(w.grad.equal(x.mm(r.t()).t()))  # W.grad = XRT  : m x n

        x = torch.tensor([[7., 9.], [11., 13.], [15., 17.]],
                         requires_grad=True)  # n x k
        w = torch.tensor([[2., -3., 4.], [40., 50., -60.]],
                         requires_grad=True)  # m x n
        y = w.mm(x)

        r = torch.randn(2, 2)
        y.backward(r)
        self.assertTrue(x.grad.equal(w.t().mm(r)))  # X.grad = WT r : n x k
        self.assertTrue(w.grad.equal(x.mm(r.t()).t()))  # W.grad = XRT  : m x n

    def test6(self):
        print()
        a = torch.tensor([1.], requires_grad=True)
        w1 = torch.tensor([1.], requires_grad=True)
        w2 = torch.tensor([1.], requires_grad=True)
        w3 = torch.tensor([1.], requires_grad=True)
        w4 = torch.tensor([1.], requires_grad=True)

        b = w1 * a
        c = w2 * a
        d = w3 * b + w4 * c
        L = d + 10

        L.backward()
        for x in [a, w1, w2, w3, w4, b, c, d, L]:
            printNode(x)

    def test7(self):
        a = torch.ones(3, dtype=torch.float, requires_grad=True)
        w1 = torch.ones((2, 3), dtype=torch.float, requires_grad=True)
        w2 = torch.ones((2, 3), dtype=torch.float, requires_grad=True)
        w3 = torch.ones((3, 2), dtype=torch.float, requires_grad=True)
        w4 = torch.ones((3, 2), dtype=torch.float, requires_grad=True)

        b = w1.mv(a)  # (2 x 3) x (3 x 1) = (2 x 1)
        c = w2.mv(a)  # (2 x 3) x (3 x 1) = (2 x 1)
        d = w3.mv(b) + w4.mv(c)  # (3 x 2) x (2 x 1) = (3 x 1)
        L = d + 1

        L.backward(torch.tensor([1., 1., 1.]))
        for x in [a, w1, w2, w3, w4, b, c, d, L]:
            printNode(x)

    def test8(self):
        eta = 0.5
        x0 = torch.ones((2, 4), dtype=torch.float)
        w1 = torch.ones((2, 3), dtype=torch.float, requires_grad=True)
        w2 = torch.ones((3, 2), dtype=torch.float, requires_grad=True)
        t = torch.zeros((2, 4), dtype=torch.float)

        x2 = w2.t().mm(w1.t().mm(x0))
        loss = torch.dist(x2, t)
        loss.backward()

        for x in [w1, w2, t, loss]:
            printNode(x)

        with torch.no_grad():
            w1 -= eta * w1.grad
            w2 -= eta * w2.grad
            w1.grad.zero_()
            w2.grad.zero_()

        x2 = w2.t().mm(w1.t().mm(x0))
        loss = torch.dist(x2, t)
        loss.backward()

        for x in [w1, w2, t, loss]:
            printNode(x)

    def test9(self):
        # cuda = torch.device("cuda:0")
        cuda = torch.device("cpu")
        tf = torch.float
        eta = 0.001
        x0 = torch.ones((2, 4), device=cuda, dtype=tf)
        w1 = torch.ones((2, 3), device=cuda, dtype=tf, requires_grad=True)
        w2 = torch.ones((3, 2), device=cuda, dtype=tf, requires_grad=True)
        t = 10 * torch.ones((2, 4), device=cuda, dtype=tf)

        for i in range(5000):

            # x1 = w1.t().mm(x0).clamp(min=0)
            x1 = torch.nn.Sigmoid()(w1.t().mm(x0))
            x2 = w2.t().mm(x1)

            loss = torch.dist(x2, t)
            loss.backward()
            if i % 500 == 0:
                print(loss)

            # for x in [w1, w2, t, loss]:
            #     printNode(x)

            with torch.no_grad():
                w1 -= eta * w1.grad
                w2 -= eta * w2.grad

                # Manually zero the gradients after updating weights
                w1.grad.zero_()
                w2.grad.zero_()

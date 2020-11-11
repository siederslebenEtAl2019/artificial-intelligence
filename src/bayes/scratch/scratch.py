class X(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.__u = 27

    def get_a(self):
        return self.a

    def get_b(self):
        return self.b

    def get_ab(self):
        raise NotImplemented

    def test(self):
        return self.get_ab()

    def make(self):
        return X(self.a, self.b)


class Y(X):
    def __init__(self, a, b, c):
        super().__init__(a, b)
        self.c = c

    def __str__(self):
        return str((self.get_a(), self.get_b(), self.get_c()))

    def p(self):
        return self.a, self.b

    def get_c(self):
        return self.c

    def get_ab(self):
        return self.a, self.b

    def make(self):
        return Y(self.a, self.b, self.c)


class Z(X):
    def __init__(self, a, b, c):
        super().__init__(a, b)
        self.c = c

    def nothing(self):
        pass


if __name__ == '__main__':
    x = X(6, 7)
    print(x)
    y = Y(2, 3, 4)
    print(y)
    print(y.test())
    print(y.p())

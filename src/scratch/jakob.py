

def poly(a, b, c):
    return lambda x: a * x ** 3 + b * x + c


def dpoly(a, b):
    return lambda x: 3 * a * x ** 2 + b


def cardan(a, b, c):
    """
    :return: real solution of f(x) = a * x^3 + b * x + c
    """
    p = b/a
    q = c/a

    delta = (q/2)**2 + (p/3)**3
    u = (- q/2 + delta ** (1/2)) ** (1/3)
    v = (- q/2 - delta ** (1/2)) ** (1/3)

    return u + v


if __name__ == '__main__':
    a = 17985.  # h^3
    b = -493.01136  # h
    c = -816069.38  # const

    z = cardan(a, b, c)
    print(z, poly(a, b, c)(z), dpoly(a, b)(z))




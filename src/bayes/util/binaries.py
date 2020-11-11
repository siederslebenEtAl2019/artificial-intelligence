import functools as ft

epsilon = 1e-5


def binary_cube(pattern):
    """
       :param: pattern is a list of 0, 1 and None
       :return: projection of binary cube of dimension = n
       [0, None, 1] -> [0, 0, 1], [0, 1, 1]
       [0, None, None] -> [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]
       """
    current = list(pattern)
    nones = [i for (i, v) in enumerate(pattern) if pattern[i] is None]
    k = len(nones)
    if k == 0:
        yield current
    else:
        for j in range(2 ** k):
            b = int2bin(j, k)
            current = current.copy()
            for i in range(k):
                current[nones[i]] = b[i]
            yield current


def int2bin(i, n):
    """
    :param i: an integer >= 0
    :param n: length of list
    :return: binary digits of i as a list of length at least n
    6, 3 -> [1, 1, 0]
    6, 4 -> [0, 1, 1, 0]
    """
    return [int(c) for c in bin(i)[2:].rjust(n, '0')]


def bin2int(b):
    """
    :param b: a list of binary digits, e.g. [0,1,0,0,1]
    :return: integer value of this list
    [0,1,0,0,1] -> 9
    """
    return ft.reduce(lambda r, d: d + (r << 1), b, 0)


def rev_int(i, n):
    """
    :param i:
    :param n:
    :return: i in reversed binary order
    This is equivalent  to
    b = int2bin(i, n)
    b.reverse()
    return bin2int(b)
    """
    b = '{:0{width}b}'.format(i, width=n)
    return int(b[::-1], 2)


def log2(n):
    """
    :param n: an integer >= 0
    :return: (number of binary digits of n) - 1
    So: log2(1) = 0, log2(2) = 1, log2(3) = 1
    """
    if n < 1:
        raise ValueError
    result = -1
    while n > 0:
        n >>= 1
        result += 1
    return result

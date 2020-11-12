
# lecture 1, 12.11.2020
# warmup

def fibo1(n):
    if n < 0:
        raise ValueError
    if n <= 1:
        return n
    return fibo1(n-2) + fibo1(n-1)


def fibo2(n):
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == '__main__':
    for fibo in [fibo1, fibo2]:
        print(fibo(3))


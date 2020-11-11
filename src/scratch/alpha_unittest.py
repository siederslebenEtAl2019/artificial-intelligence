import math, unittest


# Constants
g = 9.81  # gravity force in [m/s²]


def re(vel, dia, visc):  # program to compute Reynolds number
    """
    :param vel: velocity in [m/s]
    :param dia: diameter of the pipe in [m]
    :param visc: viscosity for water at 20°C in [m²/s]
    :return: Reynolds number
    """
    Re = vel * dia / visc
    # if Re > 2300:  # check if flow is turbulent or laminar
    #     print('The Reynoldsnumber is', Re, 'thus the flow is turbulent\n')
    # else:
    #     print('The Reynoldsnumber is', Re, 'thus the flow is laminar\n')
    return Re


def alpha(alpha_0, k, dia, Re, count):
    """
    :param alpha_0: ??
    :param k: roughness in [m] (range 0.001 (rough) - 0.0001 (smooth))
    :param dia:  diameter of the pipe in [m]
    :param Re: Reynolds number
    :param count: starting value for counting the iterations
    :return: alpha = loss coefficient for flow in pipe
    """

    while True:  # iterative loop to compute alpha
        alpha_1 = (1 / (-2 * math.log10(
            (k / (3.71 * dia)) + (2.51 / (Re * alpha_0 ** 0.5))))) ** 2  # equation to compute alpha
        diff = alpha_1 - alpha_0  # comparison between new and old alpha
        alpha_0 = alpha_1  # overwrites old alpha with new alpha
        count = count + 1  # counts the iteration steps
        if abs(diff) < 0.000000001:  # condition when to stopp the loop
            break

    # print('The value of alpha is', alpha_1, 'and it needed', count, 'iterations \n')
    return alpha_1, count


def energyloss(alpha, len, dia, vel):
    """
    :param alpha: loss coefficient for flow in pipe
    :param len: length of the pipe in [m]
    :param dia: diameter of the pipe in [m]
    :param vel: velocity in [m/s]
    :param g: gravity force in [m/s²]
    :return: energy loss
    """
    el = alpha * (len / dia) * (vel ** 2 / g)
    # print('The energyloss in the pipe is ', el, 'm')
    return el


class TestAlpha(unittest.TestCase):
    # k = 0.0005  # roughness in [m] (range 0.001 (rough) - 0.0001 (smooth))
    #     # dia = 0.5  # diameter of the pipe in [m]
    #     # vel = 1  # velocity in [m/s]
    #     # visc = 1e-6  # viscosity for water at 20°C in [m²/s]
    #     # count = 0  # starting value for counting the iterations
    #     # len = 10  # length of the pipe in [m]
    #     # alpha_0 = 0.1

    def testAlpha(self):
        test1 = {'k': 0.0005, 'dia': 0.5, 'vel': 1, 'visc': 1e-6, 'count': 0, 'len': 10, 'alpha_0': 0.1}
        test2 = {'k': 0.0006, 'dia': 0.6, 'vel': 2, 'visc': 2e-6, 'count': 1, 'len': 11, 'alpha_0': 0.2}
        test3 = {'k': 0.0007, 'dia': 0.7, 'vel': 3, 'visc': 3e-6, 'count': 2, 'len': 12, 'alpha_0': 0.3}

        result1 = [500000.0, 0.2038735983690112]
        result2 = [600000.0, 1.4950730547060824]
        result3 = [700000.0, 4.718217562254258]

        testcases = [test1, result1], [test2, result2], [test3, result3]

        for t, r in testcases:
            reynolds = re(t['vel'], t['dia'], t['visc'])
            el = energyloss(t['alpha_0'], t['len'], t['dia'], t['vel'])
            self.assertAlmostEqual(r[0], reynolds, 5)
            self.assertAlmostEqual(r[1], el, 5)


if __name__ == '__main__':
    unittest.main()
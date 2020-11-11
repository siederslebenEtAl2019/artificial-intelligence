import math

# this code calculates the energy loss coefficient for a round pipe

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


def alphadialog(k, count, visc):
    startmessage = '\n please enter a positive number for the starting value of the iteration!\n'
    velocitymessage = 'What is the velocity in [m/s] in the pipe?\n'
    diametermessage = 'What is the diameter of the pipe in [m]?\n'
    lengthmessage = 'What is the length of the pipe in [m]?\n'
    errormessage = 'please enter a positive number!\n'

    while True:
        inp = input(startmessage)
        alpha_0 = float(inp)
        if alpha_0 <= 0:
            print(errormessage)
            break
        inp = input(velocitymessage)
        vel = float(inp)
        if vel <= 0:
            print(errormessage)
            break
        inp = input(diametermessage)
        dia = float(inp)
        if dia <= 0:
            print(errormessage)
            break
        inp = input(lengthmessage)
        len = float(inp)
        if len <= 0:
            print(errormessage)
            break

        Re = re(vel, dia, visc)  # compute Reynoldsnumber

        # (source for tuples: https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/)
        alpha_1, count_1 = alpha(alpha_0, k, dia, Re, count) # compute alpha and number of iterations
        el = energyloss(alpha_1, len, dia, vel)  # compute energyloss coefficient

        print(f"{'startvalue':>15} {alpha_0:6.4} \n"
              f"{'length':>15} {len:6.4} \n"
              f"{'diameter':>15} {dia:6.4} \n"
              f"{'Reynold number':>15} {Re:8.4} \n"
              f"{'energy loss':>15} {el:11.4} \n"
              f"{'count':>15} {count_1:4}")


if __name__ == '__main__':
    k = 0.0005  # roughness in [m] (range 0.001 (rough) - 0.0001 (smooth))
    count = 0  # starting value for counting the iterations
    visc = 1e-6  # viscosity for water at 20°C in [m²/s]

    alphadialog(k, count, visc)

# j. siedersleben, QAware GmbH, Munich
# 1/4/2020
# backward propagation made easy

import time
import unittest

from modules.nnw import nnw_t3, nnw_autograd, nnw_t2, nnw_torch

import simple
import simple1
import tiny

from simple import *
from simple1 import *
from tiny import *

method_dict = {nnw_numpy.backprop: Testmanager.getArgsNumpy,
               nnw_torch.backprop: Testmanager.getArgsTorch,
               nnw_autograd.backprop: Testmanager.getArgsTorch,
               nnw_t2.backprop: Testmanager.getArgsTorch,
               nnw_t3.backprop: Testmanager.getArgsTorch}

testcases = {'T': [tiny0, tiny1, tiny2, tiny3],
             'S1': [simple10, simple11, simple12, simple13],
             'S': [simple0, simple1, simple2, simple3]}


def storeAll(s1=slice(None, None),
             s2=slice(None, None),
             s3=slice(None, None)):
    tiny.storeAll(s1)
    simple.storeAll(s2)
    simple1.storeAll(s3)


class Testnnw(unittest.TestCase):
    def testStore(self):
        storeAll()

    def testOnce(self):
        times = []
        for tc in testcases['S1'][3:4]:
            mgr = tc()
            print('\n', mgr.name)
            start = time.perf_counter()
            bp = nnw_autograd.backprop
            wstar, ctrl = bp(**method_dict[bp](mgr))
            times.append(time.perf_counter() - start)
            cstar = ctrl.lastvalue()
            print('c* = ', cstar, '\ncounter = ', ctrl.counter)
            for data in ctrl.report[-5:]:
                print(data[0], data[1])
            print('w* =')
            for v in wstar[1:4]:
                print(v[:4, :4])

            if mgr.m == 1:  # one layer only
                ws, cs = nnw_numpy.solve(mgr.x0, mgr.t)
                print('diff c* = ', cs - cstar, '\n', 'diff w* = ', abs(ws - wstar[1]).max())
        print('elapsed times: ', times)

    def testRepeat(self):
        for tc in testcases['S1'][3:4]:
            mgr = tc()
            print('\n', mgr.name)
            results = []
            times = []
            for cnt in range(3):
                start = time.perf_counter()
                bp = nnw_torch.backprop
                wstar, ctrl = bp(**method_dict[bp](mgr))
                times.append(time.perf_counter() - start)
                results.append([wstar, ctrl])
                mgr.random_w()
                print()

            print('elapsed times: ', times)
            print('counters:      ', [r[1].counter for r in results])
            ws = [r[1].lastvalue() for r in results]
            diffw = max(ws) - min(ws)
            print('\ndiffw = ', diffw)

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    ww = results[i][0]
                    vv = results[j][0]
                    diff = sum([nnw_numpy.euclid(ww[k], vv[k]) for k in range(1, len(ww))])
                    print(i, j, 'diff = ', diff)

    def testCompare(self):
        for tc in testcases['S1'][3:4]:
            results = []
            times = []
            mgr = tc()
            print('\n', mgr.name)
            for bp in [nnw_numpy.backprop, nnw_torch.backprop, nnw_t2.backprop]:
                start = time.perf_counter()
                wstar, ctrl = bp(**method_dict[bp](mgr))
                times.append(time.perf_counter() - start)
                results.append((wstar, ctrl))

            print('elapsed times: ', times)
            print('counters:      ', [r[1].counter for r in results])
            ws = [r[1].lastvalue() for r in results]
            diffw = max(ws) - min(ws)
            print('\ndiffw = ', diffw)

            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    ww = results[i][0]
                    vv = results[j][0]
                    diff = sum([nnw_numpy.euclid(ww[k], vv[k]) for k in range(1, len(ww))])
                    print(i, j, 'diff = ', diff)

# 24.08.2020


import time, torch
import unittest


def performanceIndicators(true_negatives, false_negatives, false_positives, true_positives):
    total = true_negatives + false_negatives + false_positives + true_positives
    accuracy = (true_positives + true_negatives) / total             # share of correct predictions
    precision = true_positives / (true_positives + false_positives)  # share of true positives among found positives
    recall = true_positives / (true_positives + false_negatives)     # share of found positives among all positives
    f1 = 2 * precision * recall / ( precision + recall)
    return accuracy, precision, recall, f1


class Stats(object):
    """
    Stats-objects are created with a name and an optional parent. They build an hierarchy of
    arbitrary depth. A Stats-object supports any number of start/stop-sequences. Start makes it
    active, stop inactive. Total_time is the cumulated time spent in active state.
    While active, a stats object supports any number of appends to the protocol. Each protocol entry contains a
    counter (since creation of the Stats object), a timestamp and some numerical data which can be added up
    A Stats object returns a summary at any time and any number of times. The summary returns the summary of itself
    and of all its kids.

    """
    def __init__(self, name, parent=None):
        self.name = name
        self.active = False
        self.start_time = 0
        self.total_time = 0
        self.result = [0, 0, 0, 0]
        self.kids = []
        self.parent = parent
        if parent is not None:
            parent.kids.append(self)
        self.counter = 0

    def start(self):
        if self.active:
            raise Exception
        self.active = True
        self.start_time = time.perf_counter()

    def stop(self):
        if not self.active:
            raise Exception
        stop_time = time.perf_counter()
        self.total_time += stop_time - self.start_time
        self.active = False

    def append(self, result):
        if not self.active:
            raise Exception
        self.counter += 1
        self.protocol.append((self.counter, time.perf_counter(), result))

    def summary(self):
        pt_protocol = torch.tensor(self.protocol, requires_grad=False)
        sums = torch.sum(pt_protocol[:, 2:], 0)
        avgs = sums / len(self.protocol)
        mysummary = (self.name, self.total_time, len(self.protocol), sums, avgs)
        kidssummary = [k.summary() for k in self.kids]
        return mysummary if len(self.kids) == 0 else [mysummary, kidssummary]


class TestStats(unittest.TestCase):
    def test1(self):
        s = Stats('S')
        s.start()
        s.append(3, 4, 5)
        s.append(4, 5, 6)
        s.stop()
        print(s.summary())

    def test2(self):
        s = Stats('S')
        s.start()
        t = Stats('T', s)
        t.start()
        r = Stats('R', s)
        r.start()
        s.append(3, 4, 5)
        s.append(4, 5, 6)
        t.append(30, 40, 50)
        t.append(40, 50, 60)
        r.append(300, 400, 500)
        r.append(400, 500, 600)
        r.stop()
        t.stop()
        s.stop()
        print(s.summary())

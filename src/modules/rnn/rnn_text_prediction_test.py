# from
# https://github.com/gabrielloye/RNN-walkthrough/blob/master/main.ipynb

import unittest

import torch

from modules.rnn.rnn_text_prediction import TextPredictionManager

device = torch.device("cuda:0")
# device = torch.device("cpu")


class TestTextPrediction(unittest.TestCase):

    def test1(self):
        sample = ['hey how are you', 'good i am fine', 'have a nice day']
        # sample = ['he']

        mgr = TextPredictionManager(sample)
        mgr.trainingLoop(100, 1e-2)

        from_text = ['ha']
        for s in from_text:
            guess = mgr.predict(s)
            print('\n', s, '\t', guess)

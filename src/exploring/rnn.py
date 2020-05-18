import unittest
import torch
import torch.nn as nn


class TestRNN(unittest.TestCase):

    def testRNN(self):

        batch = 1
        sequence_length = 1
        input_size = 1  # number of features, e.g. characters
        hidden_size = 1  # arbitrary
        layers = 1  # default = 1

        x_shape = [batch, sequence_length, input_size]
        y_shape = [batch, sequence_length, hidden_size]
        h_shape = [layers, batch, hidden_size]

        w_ih_shape = [hidden_size, input_size]
        w_hh_shape = [hidden_size, hidden_size]
        b_ih_shape = [hidden_size]
        b_hh_shape = [hidden_size]

        x = torch.full(x_shape, 4.)
        h0 = torch.full(h_shape, 5.)

        w_ih = torch.full(w_ih_shape, 2.)
        w_hh = torch.full(w_hh_shape, 3.)
        b_ih = torch.full(b_ih_shape, 300.)
        b_hh = torch.full(b_hh_shape, 400.)

        xd = {'weight_ih_l0': w_ih,
              'weight_hh_l0': w_hh,
              'bias_ih_l0': b_ih,
              'bias_hh_l0': b_hh}

        m = nn.RNN(input_size, hidden_size, layers, batch_first=True,
                   bias=True, nonlinearity='relu')
        m.load_state_dict(xd, strict=True)

        y, h1 = m(x, h0)
        self.assertEqual(list(y.size()), y_shape)
        self.assertEqual(list(h1.size()), h_shape)

        print('\n', 'x = ', x.size(), '\n', x)
        print('\n', 'h0 = ', h0.size(), '\n', h0)
        print('\n', 'y = ', y.size(), '\n', y)
        print('\n', 'h1 = ', h1.size(), '\n', h1)

        for k, v in m.state_dict().items():
            print('\n', k, v.size(), '\n', v)

        yy = x[0].mm(w_ih.t()) + b_ih + h0[0].mm(w_hh.t()) + b_hh
        hh = x[0].mm(w_ih.t()) + b_ih + h0[0].mm(w_hh.t()) + b_hh

        print('\n', 'yy = ', yy.size(), '\n', yy)

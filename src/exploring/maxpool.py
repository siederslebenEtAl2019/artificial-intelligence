import unittest

import torch

# length_of_minibatch = 1
# width_of_kernel = 2
# width_of_input = 4
# width_of_output = width_of_input / width_of_kernel
# nbr_of_in_channels = 2
# nbr_of_ot_channels = 2


device = torch.device("cuda:0")

class Testmaxpool(unittest.TestCase):
    def test1(self):
        m = torch.nn.MaxPool1d(3, stride=2, padding=1,
                               dilation=1, return_indices=True, ceil_mode=False)

        input = torch.randn([1, 2, 5])
        output, indices = m(input)

        print('\n', input.shape, '\n', input)
        print('\n', output.shape, '\n', output)

    def testCuda(self):
        t = torch.ones(2,2)
        s = t.to(device)
        pass  # XXXXXXXXXXXXXXXXXXXXXXXXXXXX


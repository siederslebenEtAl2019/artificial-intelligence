import unittest

import torch

# m = torch.nn.Conv1d(
# in_channels,
# out_channels,
# kernel_size,
# stride=1,
# padding=0,
# dilation=1,
# groups=1,
# bias=None,
# padding_mode='zeros')  -> function m

# Conv1d.weight to be set; type = torch.nn.Parameter

# m(input) -> Tensor


# torch.nn.functional.conv1d(
# input,      input tensor of shape (length_of_minibatch, nbr_of_in_channels, width_of_input)
# weight,     filters of shape (nbr_of_out_channels, nbr_of_in_channels/groups, width_of_kernel)
#             width_of_kernel <= width_of input
#             width_of_output = floor((width_of input - width_of_output + 1) / stride)
# bias=None,  optional bias of shape (out_channels)
# stride=1,
# padding=0,
# dilation=1,
# groups=1) → output
# output.shape = (length_of_minibatch, nbr_of_out_channels, width_of output)


class Testconvolution(unittest.TestCase):
    def test0(self):
        m = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1,
                            padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        input = torch.ones(1, 1, 10)
        weight = torch.tensor([[[1., 1.]]])   # shape = (1, 1, 2)
        m.weight = torch.nn.Parameter(weight, requires_grad=True)
        output = m(input)
        print()
        print(input.shape)
        print(output.shape)
        print(output.data, '\n')
        for n, p in m.named_parameters():
            print(n, p.data)


    def testA(self):
        device = torch.device('cpu')
        # device = torch.device('cuda:0')
        minibatch = 1
        in_channels = 3
        out_channels = 1
        input_size = 4
        kernel_size = 2
        stride = 1
        padding = 0

        input_shape = [minibatch, in_channels, input_size]
        output_size = 1 + (input_size + 2 * padding - kernel_size) // stride
        output_shape = [minibatch, out_channels, output_size]
        weight_shape = [out_channels, in_channels, kernel_size]
        bias_shape = [out_channels]

        input = torch.ones(input_shape, device=device) * 1.
        weight = torch.ones(weight_shape, device=device) * 2.
        bias = torch.ones(bias_shape, device=device) * 7.

        m = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=1, groups=1, bias=True, padding_mode='zeros')
        m.to(device)
        m.weight = torch.nn.Parameter(weight, requires_grad=False)
        m.bias = torch.nn.Parameter(bias, requires_grad=False)
        output = m(input)    # minibatch = input.size[0], input_size = input.size[2]

        self.assertEqual(output_shape, list(output.shape))
        print('\n', output)

    def test1(self):
        length_of_minibatch = 5
        nbr_of_in_channels = 1
        nbr_of_out_channels = 3
        width_of_input = 10
        width_of_kernel = 3
        stride = 1
        padding = 0
        width_of_output = 1 + (width_of_input + 2 * padding - width_of_kernel) // stride

        shape_of_input = [length_of_minibatch, nbr_of_in_channels, width_of_input]
        shape_of_output = [length_of_minibatch, nbr_of_out_channels, width_of_output]

        input = torch.ones(shape_of_input)
        weight = torch.tensor([[[1., 1., 1.]], [[2., 2., 2.]], [[3., 3., 3.]]])   # shape = (3, 1, 3)

        output = torch.nn.functional.conv1d(input, weight, bias=None, stride=stride,
                                            padding=padding, dilation=1, groups=1)

        self.assertEqual(shape_of_output, list(output.shape))
        print('\n', shape_of_output, output.shape)
        print('\n', output.data)

    def test2(self):
        # two in_channels, one out_channel
        length_of_minibatch = 3
        nbr_of_in_channels = 2
        nbr_of_out_channels = 1
        width_of_input = 10
        width_of_kernel = 3
        stride = 1
        padding = 0
        width_of_output = 1 + (width_of_input + 2 * padding - width_of_kernel) // stride

        shape_of_input = [length_of_minibatch, nbr_of_in_channels, width_of_input]
        shape_of_output = [length_of_minibatch, nbr_of_out_channels, width_of_output]

        input = torch.ones(shape_of_input)
        weight = torch.tensor([[[.1, .5, .9], [.4, .8, .1]]])  # shape = (1, 2, 3)
        output = torch.nn.functional.conv1d(input, weight, bias=None, stride=stride,
                                            padding=padding, dilation=1, groups=1)

        # self.assertEqual(shape_of_output, list(output.shape))
        print('\n', shape_of_output, output.shape)
        print('\n', output)

    def test3(self):
        # two in_channels, two out_channels
        length_of_minibatch = 3
        nbr_of_in_channels = 2
        nbr_of_out_channels = 2
        width_of_input = 10
        width_of_kernel = 3
        stride = 1
        padding = 0
        width_of_output = 1 + (width_of_input + 2 * padding - width_of_kernel) // stride

        shape_of_input = [length_of_minibatch, nbr_of_in_channels, width_of_input]
        shape_of_output = [length_of_minibatch, nbr_of_out_channels, width_of_output]

        input = torch.ones(shape_of_input)
        weight = torch.tensor([[[.1, .5, .9], [.2, .6, .0]], [[.4, .8, .1], [.5, .9, .2]]])  # shape = (2, 2, 3)
        output = torch.nn.functional.conv1d(input, weight, bias=None, stride=stride,
                                            padding=padding, dilation=1, groups=1)

        # self.assertEqual(shape_of_output, list(output.shape))
        print('\n', shape_of_output, output.shape)
        print('\n', output)

    def test4(self):
        m = torch.nn.Conv1d(16, 33, 3, stride=2)
        input = torch.randn(20, 16, 50)
        output = m(input)
        print('\n', output)

import math

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear:

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weight = torch.randn(
            out_size, in_size, device=device) / math.sqrt(in_size)

        self.bias = torch.randn(
            1, self.out_size, device=device) / math.sqrt(in_size)

        self.output = None
        self.gradInput = None

    def forward(self, input):

        self.output = (torch.mm(self.weight, input.t())).t() + self.bias

        return self.output

    def backward(self, input, gradOutput, alpha=0.001):
        self.gradInput = torch.mm(gradOutput, self.weight)
        gradB = sum(gradOutput).reshape(self.bias.shape)
        gradW = torch.mm(gradOutput.t(), input.to(device))
        self.bias -= alpha * gradB
        self.weight -= alpha * gradW

        return self.gradInput


# x = torch.randn(1, 10)
# y = Linear(10, 5).forward(x)
# print(y)

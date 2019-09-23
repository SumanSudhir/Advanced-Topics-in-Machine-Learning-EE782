import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear:

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weight = torch.randn(
            out_size, in_size, dtype=torch.double, device=device)
        self.bias = torch.randn(
            self.out_size, 1, dtype=torch.double, device=device)
        self.output = None
        self.gradInput = None

    def forward(self, input, weightt=None):
        if weightt is not None:
            self.bias = torch.ones(
                self.out_size, 1, dtype=torch.double, device=device)

            self.output = torch.mm(weightt, input) + self.bias

        else:
            self.output = torch.mm(self.weight, input) + self.bias

        return self.output

    # def backward(self, input, gradOutput, alpha=0.001):
    #     self.gradInput =

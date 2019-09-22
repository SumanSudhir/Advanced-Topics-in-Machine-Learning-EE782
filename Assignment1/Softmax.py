import numpy
import torch


class Softmax:

    def __init__(self):
        self.output = output
        self.gradInput = None

    def forward(self, input):
        self.output = torch.exp(
            input) / torch.sum(torch.exp(input), dim=0)   # dim = 0 to sum all element of same column
        return self.output

    def backward(self, input, gradOutput, alpha=None):
        self.gradInput = self.gradOutput.clone()
        self.gradInput[input] = self.forward(input) * (1 - self.forward(input))
        return self.gradInput

import numpy
import torch


class Softmax:

    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input):
        self.output = torch.exp(
            input) / torch.sum(torch.exp(input), dim=1)   # dim = 0 to sum all element of same column
        return self.output

    def backward(self, input, gradOutput, alpha=None):
        self.gradInput = gradOutput.clone()
        self.gradInput[input] = self.forward(input) * (1 - self.forward(input))
        sum_of_rows = torch.sum(torch.exp(input), dim=1)

        self.gradInput[input] = (
            torch.exp(input) * (sum_of_rows - torch.exp(input))) / sum_of_rowss**2

        return self.gradInput


# x = torch.randn(1, 10)
# y = Softmax().forward(x)
# print(torch.sum(y, dim=1))

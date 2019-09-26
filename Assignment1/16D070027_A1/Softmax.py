import numpy
import torch


class Softmax:

    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input):
        # Get the probability from the score of each class
        self.output = torch.exp(
            input) / torch.sum(torch.exp(input), dim=1)   # dim = 0 to sum all element of same column
        return self.output

    def backward(self, input, gradOutput, alpha=None):
        # cloning the incoming gradient from i+1 th layer
        self.gradInput = gradOutput.clone()
        self.gradInput[input] = self.forward(input) * (1 - self.forward(input))
        # sum of exp score of each row
        sum_of_rows = torch.sum(torch.exp(input), dim=1)
        # gradient of sigmoid wrt ith input is prob[ith_input]*(1 - probability[ith_input])
        self.gradInput[input] = (
            torch.exp(input) * (sum_of_rows - torch.exp(input))) / sum_of_rowss**2

        return self.gradInput


# x = torch.randn(1, 10)
# y = Softmax().forward(x)
# print(torch.sum(y, dim=1))

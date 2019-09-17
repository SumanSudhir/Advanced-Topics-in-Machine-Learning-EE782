import numpy
import torch


class Softmax:

    def __init__(self):
        self.output = output

    def forward(self, input):
        self.output = torch.exp(
            input) / torch.sum(torch.exp(input), dim=0)
        return self.output

    def backward(self, input):

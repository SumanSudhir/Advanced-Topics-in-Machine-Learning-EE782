import torch


class ReLu:

    def __init__(self):
        self.output = None

    def forward(self, input):
        self.output = input.clamp(0)
        return self.output

    # def backward(self, input, grad)

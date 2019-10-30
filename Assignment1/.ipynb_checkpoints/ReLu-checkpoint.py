import torch


class ReLu:

    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input):
        self.output = input.clamp(0)
        return self.output

    def backward(self, input, gradOutput, alpha=None):
        self.gradInput = gradOutput.clone()
        self.gradInput[input < 0] = 0.0
        return self.gradInput

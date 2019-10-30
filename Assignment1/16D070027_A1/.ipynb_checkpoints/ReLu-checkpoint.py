import torch


class ReLu:

    def __init__(self):
        self.output = None
        self.gradInput = None

    def forward(self, input):
        # clamp the output when input as max(0,x)
        self.output = input.clamp(0)
        return self.output

    def backward(self, input, gradOutput, alpha=None):
        # clone the gradOutput of i+1 layer
        self.gradInput = gradOutput.clone()
        # Derivate of relu wrt to input is 1 when x>0 and 0 else it will remain same
        self.gradInput[input < 0] = 0.0
        return self.gradInput

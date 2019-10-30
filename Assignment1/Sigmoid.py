import torch


class Sigmoid:

    def __init__(self):
        self.gradInput = None

    def forward(self, input):
        return 1 / (1 + torch.exp(-input))

    # def backward(self, input, gradOutput, alpha=None):
    #     self.gradInput = gradOutput.clone()

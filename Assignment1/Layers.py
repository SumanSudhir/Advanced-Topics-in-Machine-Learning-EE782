import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Layers:

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        self.weight = torch.randn(
            out_size, in_size, dtype=torch.double, device=device)
        self.bias = torch.randn(out_size, 1, dtype=torch.double, device=device)
        self.output = None

    def forward(self, input):
        self.output = torch.mm(self.weight, input) + self.bias
        return self.output

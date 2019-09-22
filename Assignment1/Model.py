import torch

from Layers import Layers

input = torch.tensor([[0.1], [0.2], [0.7]], dtype=torch.double)
hidden1 = Layers(3, 3).forward(input)
print(hidden1)

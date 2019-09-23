import torch

from Cross_entropy import CrossEntropy
from Linear import Linear
from ReLu import ReLu
from Sigmoid import Sigmoid
from Softmax import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# input = torch.tensor([[0.1], [0.2], [0.7]], dtype=torch.double)
# weight1 = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.2, 0.7], [
#                        0.4, 0.3, 0.9]], dtype=torch.double)
#
# weight2 = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.5, 0.7], [
#                        0.6, 0.4, 0.8]], dtype=torch.double)
#
# weight3 = torch.tensor([[0.1, 0.4, 0.8], [0.3, 0.7, 0.2], [
#                        0.5, 0.2, 0.9]], dtype=torch.double)
#
#
# # print(weight1.shape)
# # print(input.shape)
# hidden1 = Layers(3, 3).forward(input, weight1.t())
# hidden1 = ReLu().forward(hidden1)
# # print(hidden1)
#
# hidden2 = Layers(3, 3).forward(hidden1, weight2.t())
# hidden2 = Sigmoid().forward(hidden2)
# # print(hidden2)
#
# hidden3 = Layers(3, 3).forward(hidden2, weight3.t())
# loss = CrossEntropy().forward(hidden3, torch.tensor([2]).t())
# # print(hidden3)
# print(loss)

print(device)


class Model:

    def __init__(self):
        self.layers = []
        self.isTrain = False  # While training make it true

    def forward(self, input):
        output = input.clone().to(device)
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, input, gradOutput, alpha=None):
        for i in range(len(self.layers) - 1, 0, -1):
            gradOutput = self.layers[i].backward(
                self.layers[i - 1].output, gradOutput, alpha)

        self.layers[0].backward(input, gradOutput, alpha)

    def addLayer(self, class_object):
        self.layers.append(class_object)

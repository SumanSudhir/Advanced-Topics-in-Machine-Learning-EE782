import torch

from Cross_entropy import CrossEntropy
from Linear import Linear
from ReLu import ReLu
from Softmax import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Model:

    def __init__(self):
        # store the different layers which will be used in model
        self.layers = []

    def forward(self, input):
        # clone in order to avoid altering of data
        output = input.clone().to(device)
        # forward pass the input through all the layers
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, input, gradOutput, alpha=None):
        # backprop from layer n to 1
        for i in range(len(self.layers) - 1, 0, -1):
            # calculate the grad and pass it to i-1 layer in next cycle
            # first grad will come from the cross entropy loss
            gradOutput = self.layers[i].backward(
                self.layers[i - 1].output, gradOutput, alpha)

        # since layer -1 output is error therefore we need to
        # calculate it saperatel as the input to 1st layer is image of size 32x32x3
        self.layers[0].backward(input, gradOutput, alpha)

        # print(len(self.layers))
    # class to add layer in Model
    def addLayer(self, class_object):
        self.layers.append(class_object)

import math

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Linear:

    def __init__(self, in_size, out_size):
        # input size
        self.in_size = in_size
        # output size
        self.out_size = out_size
        # weight initilization with normalization
        self.weight = torch.randn(
            out_size, in_size, device=device) / math.sqrt(in_size)
        # bias initilization with normalization
        self.bias = torch.randn(
            1, self.out_size, device=device) / math.sqrt(in_size)

        self.output = None
        # the gradient till this layer it serve as input to i-1 th layer during backprop
        self.gradInput = None

    def forward(self, input):
        # output = weight*input + bias
        self.output = (torch.mm(self.weight, input.t())).t() + \
            self.bias

        return self.output

    def backward(self, input, gradOutput, alpha=0.001):
        # Pass the gradient to next layer.The derivative wrt input will be
        # weight therefore gradient to next layer will be gradOutput*weight of current layer
        self.gradInput = torch.mm(gradOutput, self.weight)
        # Bias have no effect on output through it as derivative is 1
        # GradW will be 1*gradOutput taking sum over batch and reshaping it to shape of bias
        gradB = sum(gradOutput).reshape(self.bias.shape)
        # GradW will be gradient of i+1 th layer * input to this layer
        gradW = torch.mm(gradOutput.t(), input.to(device))
        # Updating bias with learning rate alpha
        self.bias -= alpha * gradB
        # Updating weight with learning rate alpha
        self.weight -= alpha * gradW

        return self.gradInput


# x = torch.randn(1, 10)
# y = Linear(10, 5).forward(x)
# print(y)

import numpy as np


class Linear:

    def __init__(self,in_size=784,out_size=10):
        self.in_size = in_size
        self.out_size = out_size
        self.weight = np.random.randn(out_size, in_size)/np.sqrt(in_size)
        self.bias = np.random.randn(1,out_size)/np.sqrt(in_size)
        self.output = None


    def forward(self,input):
        self.output = (np.matmul(self.weight,input.T)).T + self.bias

        return self.output

    def backward(self,input,gradOutput,alpha=0.001, weight_decay=1e-4):
        gradB = gradOutput.reshape(self.bias.shape)
        gradW = np.matmul(gradOutput.T, input)

        self.bias -= alpha * gradB
        self.weight -= alpha * gradW + weight_decay*self.weight

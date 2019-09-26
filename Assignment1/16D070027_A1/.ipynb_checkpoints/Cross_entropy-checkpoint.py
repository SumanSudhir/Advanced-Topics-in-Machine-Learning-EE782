import torch


class CrossEntropy:

    def __init__(self):
        self.output = None

    def forward(self, input, target):
        # Extract number of labels in batch
        size = target.shape[0]
        # find the softmax probability of each classes
        prob = torch.exp(input) / torch.sum(torch.exp(input), dim=1)
        # take log with one hot encoding
        log_softmax = -torch.log(prob[range(size), target])

        # print(log_softmax)
        loss = torch.sum(log_softmax) / size  # Average loss on batch
        return loss

    def backward(self, input, target):
        # number of labels
        size = target.size()[0]
        # softmax probability of input with one-hot encoding
        prob = torch.exp(input) / torch.sum(torch.exp(input), dim=1)
        # derivative of loss wrt to input is prob - 1
        prob[range(size), target] -= 1

        return prob


# input = torch.randn(10, 1)
# target = torch.tensor([4], dtype=int)
# loss = CrossEntropy().forward(input, target)
# print(loss)
# print(prob)

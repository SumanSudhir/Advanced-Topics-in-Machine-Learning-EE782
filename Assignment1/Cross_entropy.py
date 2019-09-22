import torch


class CrossEntropy:

    def __init__(self):
        self.output = None

    def forward(self, input, target):
        size = target.size()[0]
        # total_sum = torch.sum(torch.exp(input), dim=0)
        prob = torch.exp(input) / torch.sum(torch.exp(input), dim=0)
        log_softmax = -torch.log(prob[target, range(size)])
        # print(log_softmax)
        loss = torch.sum(log_softmax) / size
        return loss

    def backward(self, input, target):
        size = target.size()[0]
        prob = torch.exp(input) / torch.sum(torch.exp(input), dim=0)
        prob[target, range(size)] -= 1

        return prob


# input = torch.randn(10, 2)
# target = torch.tensor([4, 6], dtype=int)
# loss = CrossEntropy().forward(input, target)
# prob = CrossEntropy().backward(input, target)
# print(loss)
# print(prob)

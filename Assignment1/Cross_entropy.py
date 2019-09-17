import torch


class CrossEntropy:

    def __init__(self):
        self.output = None

    def forward(self, input, target):
        exp_prob = torch.exp(input)
        sum_of_batch = exp_prob.sum(1)

        loss = 0
        loss = loss + torch.log(sum_of_batch).sum()

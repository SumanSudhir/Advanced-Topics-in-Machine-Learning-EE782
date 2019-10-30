import argparse
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchvision
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=1)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model_one = torch.load("model1epoch9999")

correct = 0
total = 0
for data in testloader:
    images, labels = data
        
    total = total + 1
    images = images.view(images.size(0), -1)

    _, indices = model_one.forward(images).max(1)

    if labels.to(device) == indices:
        correct = correct + 1

print("Accuracy", correct / total)

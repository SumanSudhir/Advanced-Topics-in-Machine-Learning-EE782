# %matplotlib inline
# %config InlineBackend.figure_format = retina
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from Cross_entropy import CrossEntropy
from Linear import Linear
from Model import Model
from ReLu import ReLu
from Sigmoid import Sigmoid
from Softmax import Softmax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()

my_model = Model()
my_model.addLayer(Linear(3072, 2048))
my_model.addLayer(ReLu())

my_model.addLayer(Linear(2048, 1024))
my_model.addLayer(ReLu())

my_model.addLayer(Linear(1024, 512))
my_model.addLayer(ReLu())

my_model.addLayer(Linear(512, 10))
my_model.addLayer(CrossEntropy())

print(my_model)

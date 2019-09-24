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

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    my_model = Model()
    my_model.addLayer(Linear(3072, 1024))
    my_model.addLayer(ReLu())

    # my_model.addLayer(Linear(2048, 1024))
    # my_model.addLayer(ReLu())

    my_model.addLayer(Linear(1024, 512))
    my_model.addLayer(ReLu())

    #
    my_model.addLayer(Linear(512, 2))
    # my_model.addLayer(Softmax())
    # my_model.addLayer(CrossEntropy())

    running_loss = 0
    epochs = 7
    train_count = 0
    train_losses, test_losses = [], []
    i = 0
    for epoch in range(epochs):

        for images, labels in trainloader:

            if train_on_gpu:
                images, labels = images.to(device), labels.to(device)

#             print(labels)
            if labels == 0 or labels == 1:
                train_count += 1
                
                images = images.view(images.size(0), -1)

                final_prob = my_model.forward(images)
                backward_grad = CrossEntropy().backward(final_prob, labels)
                # print(backward_grad)
                my_model.backward(images, backward_grad, alpha=0.001)

                running_loss += (CrossEntropy().forward(final_prob, labels))

            if (train_count + 1) % 500 == 0:
                i = i + 1
                test_loss = 0
                accuracy = 0
                correct_class = 0
                test_count = 0

                for images, labels in testloader:
                    if train_on_gpu:
                        images, labels = images.to(device), labels.to(device)

                    if labels == 0 or labels == 1:

                        test_count += 1

                        images = images.view(images.size(0), -1)

                        score = my_model.forward(images)
                        test_loss += CrossEntropy().forward(score, labels)

                        ps = torch.exp(score)
                        top_p, top_class = ps.topk(1, dim=1)

                        if top_class == labels:
                            correct_class += 1

                train_losses.append(running_loss / (train_count + 1))
                test_losses.append(test_loss / (test_count + 1))

#                 plt.plot(train_losses, label='Training loss')
#                 plt.plot(test_losses, label='Validation loss')
#                 plt.savefig('myfilename.png', dpi=100)

                print(f"Epoch {i}.. "
                      f"Train loss: {running_loss/(train_count):.3f} .."
                      f"Test loss: {test_loss/(test_count + 1):.3f} .."
                      f"Test accuracy: {correct_class/(test_count + 1):.3f}")

                test_count = 0
                train_count = 0
                running_loss = 0
                
        
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.savefig('final.png', dpi=100)

    return my_model


my_model = train_model()

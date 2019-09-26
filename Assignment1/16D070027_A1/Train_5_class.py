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
from Softmax import Softmax

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model():
    # Transform the image by normalizing it
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Download Training data
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # Download Test data
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    # Make trainloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=True)

    # Make testloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=1)

    # Class present in training and test data
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    """
    # Function to display Image
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    """
    # My model
    my_model = Model()
    my_model.addLayer(Linear(3072, 1024))
    my_model.addLayer(ReLu())

    # my_model.addLayer(Linear(2048, 1024))
    # my_model.addLayer(ReLu())

    my_model.addLayer(Linear(1024, 512))
    my_model.addLayer(ReLu())

    my_model.addLayer(Linear(512, 5))
    # my_model.addLayer(Softmax())
    # my_model.addLayer(CrossEntropy())

    # Loop to train the Model
    running_loss = 0
    # Number of epochs
    epochs = 7
    train_count = 0
    train_losses, test_losses = [], []
    train_correct = 0
    i = 0
    for epoch in range(epochs):

        for images, labels in trainloader:

            # Transfer it to GPU
            if train_on_gpu:
                images, labels = images.to(device), labels.to(device)

            if labels <= 4:
                # To count number to training image in each epoch
                train_count += 1
                # Flatteing of image to bring it to size batch_sizex(32*32*3)
                images = images.view(images.size(0), -1)
                # forward the image through the model
                final_prob = my_model.forward(images)
                # Calculate the backward gradient of CrossEntropy
                backward_grad = CrossEntropy().backward(final_prob, labels)
                # changing in to exp score
                ps = torch.exp(final_prob)
                # getting the top class
                top_p, top_class = ps.topk(1, dim=1)

                if top_class == labels:
                    train_correct += 1

                # Backpropagate the model
                my_model.backward(images, backward_grad, alpha=0.001)
                # calculate the running loss
                running_loss += (CrossEntropy().forward(final_prob, labels))

            # Function to Calculate Validation loss and accuracy on Validation data
            if (train_count + 1) % 500 == 0:
                i = i + 1
                test_loss = 0
                correct_class = 0
                test_count = 0

                for images, labels in testloader:
                    if train_on_gpu:
                        images, labels = images.to(device), labels.to(device)

                    if labels <= 4:

                        test_count += 1
                        # Flatteing of image
                        images = images.view(images.size(0), -1)

                        # forward the image in trained model
                        score = my_model.forward(images)
                        # calculate loss
                        test_loss += CrossEntropy().forward(score, labels)
                        # selct the top class with max score
                        ps = torch.exp(score)
                        top_p, top_class = ps.topk(1, dim=1)
                        # if top_class is same as the target label than increse correct count by 1
                        if top_class == labels:
                            correct_class += 1

                # Append to plot graph
                train_losses.append(running_loss / (train_count + 1))
                test_losses.append(test_loss / (test_count + 1))

                print(f"Epoch {i}.. "
                      f"Train loss: {running_loss/(train_count):.3f} .."
                      f"Test loss: {test_loss/(test_count):.3f} .."
                      f"Train accuracy: {train_correct/(train_count):.3f}.."
                      f"Test accuracy: {correct_class/(test_count):.3f}")

                train_correct = 0
                train_count = 0
                running_loss = 0

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    #plt.savefig('mlp2.png', dpi=100)

    return my_model


my_model = train_model()

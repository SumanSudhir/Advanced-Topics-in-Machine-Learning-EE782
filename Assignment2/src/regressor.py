import numpy as np
from Linear import Linear

def cross_entropy(input,target):
    prob = np.exp(input)/np.sum(np.exp(input), axis=1)
    loss = -np.log(prob[0,target])

    return loss

# def softmax(input):
#     return np.exp(input)/np.sum(np.exp(input), axis=1)


################################################################################
"""Training and Testing of Regressor Function"""
import pickle
from struct import unpack
import gzip
np.random.seed(2)
Linear = Linear()
img_num = 60000
def train(x_train,y_train,alpha=0.001,epochs=40):
    for i in range(epochs):
        train_loss = 0
        count = 0
        for j in range(img_num):
            input = x_train[j].copy().reshape(1,784)/255.0
            target = y_train[j].copy()
            #print(target)
            #Forward
            linear = Linear.forward(input)
            loss = cross_entropy(linear, target)
            train_loss += loss
            #Backward
            #print(loss.shape)
            prob = np.exp(linear)/np.sum(np.exp(linear), axis=1)
            prob[0,target] -= 1
            Linear.backward(input,prob,alpha)
            #Training Accuracy
            train_prob = linear
            if np.argmax(train_prob) == target:
                count += 1

        #Testing Accuracy
        num_test = 10000
        test_count = 0
        for k in range(num_test):
            test_input = x_test[k].copy().reshape(1,784)/255.0
            test_target = y_test[k].copy()
            test_prob = Linear.forward(test_input)

            if np.argmax(test_prob) == test_target:
                test_count += 1


        print("Epoch",i," ", "Loss", " ", train_loss/img_num, "Train_Accuracy"," ", count/img_num, "Test_Accuracy", " ", test_count/num_test)


#Dataset
f = gzip.open("../data/mnist.pkl.gz", "rb")
data = pickle.load(f,encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data

#print(x_train[100].reshape(1,784))
train(x_train, y_train)

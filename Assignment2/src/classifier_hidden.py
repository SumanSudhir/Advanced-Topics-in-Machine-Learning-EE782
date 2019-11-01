import numpy as np
from Linear import Linear
from RBM import RBM

def cross_entropy(input,target):
    prob = np.exp(input)/np.sum(np.exp(input), axis=1)
    loss = -np.log(prob[0,target])

    return loss

################################################################################
"""Training and Testing of Regressor Function"""
import pickle
from struct import unpack
import gzip
np.random.seed(2)
Linear = Linear(in_size=144,out_size=10)

#calling model for creating features
model_test = RBM(n_visible=28*28, n_hidden=144)

#Loading Weight
rbm_weight_file = open("model/rbm_weight.npy",'rb')
model_test.weight = pickle.load(rbm_weight_file)

#Loading visible layer bias
rbm_v_bias_file = open("model/rbm_v_bias.npy",'rb')
model_test.v_bias = pickle.load(rbm_v_bias_file)

#Loading hidden layer bias
rbm_h_bias_file = open("model/rbm_h_bias.npy",'rb')
model_test.h_bias = pickle.load(rbm_h_bias_file)


img_num = 60000
def train(x_train,y_train,alpha=0.001,weight_decay=0.0,epochs=40):
    for i in range(epochs):
        train_loss = 0
        count = 0
        for j in range(img_num):
            input = x_train[j].copy().reshape(1,784)/255.0
            target = y_train[j].copy()

            #Extracting hidden features using RBM
            input, _ = model_test.sample_h_given_v(input)

            #Forward
            linear = Linear.forward(input)
            loss = cross_entropy(linear, target)
            train_loss += loss

            #Backward
            prob = np.exp(linear)/np.sum(np.exp(linear), axis=1)
            prob[0,target] -= 1
            Linear.backward(input,prob,alpha,weight_decay)
            #Training Accuracy
            train_prob = np.exp(linear)
            if np.argmax(train_prob) == target:
                count += 1

        #Testing Accuracy
        num_test = 10000
        test_count = 0
        for k in range(num_test):
            test_input = x_test[k].copy().reshape(1,784)/255.0
            test_input,_ = model_test.sample_h_given_v(test_input)
            test_target = y_test[k].copy()
            test_prob = Linear.forward(test_input)
            test_prob = np.exp(test_prob)

            if np.argmax(test_prob) == test_target:
                test_count += 1


        print("Epoch",i," ", "Loss", " ", train_loss/img_num, "train_Acc"," ", count/img_num, "test_Acc", " ", test_count/num_test)

    classifier_weight_rbm = open(b"model/classifier_weight_rbm.npy","wb")
    pickle.dump(Linear.weight,classifier_weight_rbm)

    classifier_bias_rbm = open(b"model/classifier_bias_rbm.npy","wb")
    pickle.dump(Linear.bias,classifier_bias_rbm)

#Dataset
f = gzip.open("../data/mnist.pkl.gz", "rb")
data = pickle.load(f,encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data

#print(x_train[100].reshape(1,784))
train(x_train, y_train)

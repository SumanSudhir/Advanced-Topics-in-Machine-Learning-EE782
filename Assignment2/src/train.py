import pickle
from struct import unpack
import gzip
import numpy as np

from RBM import RBM

#Dataset
f = gzip.open("../data/mnist.pkl.gz", "rb")
data = pickle.load(f,encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data
#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
# x = x_train[0].reshape(1,784)


"""Training"""
model = RBM(n_visible=28*28, n_hidden=200,lr=0.1)

training_epochs = 50
batch_size = 20
for epoch in range(training_epochs):
    # index = 0
    batch_cost = 0
    cost = 0
    for i in range(100):
        x = x_train[i].reshape(1,784)
        cost += model.const_divergence(x, K=5)

    cost = cost/100
    print('Training epoch %d, cost is ' % epoch, cost)

# end_time = time.clock()
# pretraining_time = (end_time - start_time)

# print ('Training took %f minutes' % (pretraining_time / 60.))

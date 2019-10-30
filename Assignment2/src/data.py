import numpy as np
import pickle
from struct import unpack
import gzip

f = gzip.open("../data/mnist.pkl.gz", "rb")
data = pickle.load(f,encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data
print(y_test)

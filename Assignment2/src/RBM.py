import numpy as np

"""Implementation of Restricted Boltzmann Machine(RBM)"""
class RBM():

    def __init__(self, n_visible, n_hidden):
        #Number of visible nodes
        self.n_visible = n_visible
        #Number of hidden nodes
        self.n_hidden = n_hidden
        #Weight initilization
        self.weight = np.random.randn(self.n_visible, self.n_hidden)
        #Bias initilization
        self.v_bias = np.random.randn(1, self.n_visible)
        self.h_bias = np.random.randn(1, self.n_hidden)

    def sample_h_given_v(self, x):
        temp_out = np.matmul(x, self.weight) + self.h_bias
        prob_h_given_v = sigmoid(temp_out)

        return prob_h_given_v

    def sample_v_given_h(self, y):
        temp_out = np.matmul(y, self.weight.T) + self.v_bias
        prob_v_given_h = sigmoid(temp_out)

        return prob_v_given_h


    def training(self, visible_0, visible_k, prob_h0, prob_hk):
        self.weight += np.matmul(visible_0.T, prob_h0) - np.matmul(visible_k.T, prob_hk)
        self.v_bias += np.sum((visible_0 - visible_k), 0)
        self.h_bias += np.sum((prob_h0 - prob_hk), 0)






def sigmoid(input):
    return 1.0/(1 + np.exp(-input))

import numpy as np


class RBM():
    """Implementation of Restricted Boltzmann Machine(RBM)"""

    def __init__(self, n_visible, n_hidden, lr=0.001):
        #Number of visible nodes
        self.n_visible = n_visible
        #Number of hidden nodes
        self.n_hidden = n_hidden
        #Weight initilization
        self.weight = np.random.randn(self.n_visible, self.n_hidden)
        #Bias initilization
        self.v_bias = np.random.randn(1, self.n_visible)
        self.h_bias = np.random.randn(1, self.n_hidden)
        self.lr = lr
#         self.cost = 0

    def sample_h_given_v(self, x):
        temp_out = np.matmul(x, self.weight) + self.h_bias
        prob_h_given_v = sigmoid(temp_out)

        return prob_h_given_v, np.random.binomial(1,prob_h_given_v)

    def sample_v_given_h(self, y):
        temp_out = np.matmul(y, self.weight.T) + self.v_bias
        prob_v_given_h = sigmoid(temp_out)

        return prob_v_given_h, np.random.binomial(1,prob_v_given_h)

    def const_divergence(self, visible, K=1):
        """Implementation of Constructive divergence using gibbs sampling"""

        batch_size = np.shape(visible)[0]
        prob_h_given_v, h_sample = self.sample_h_given_v(visible)

        #positive divergence
        positive_div = np.matmul(visible.T, prob_h_given_v)

        prob_v_given_hk,_ = self.sample_v_given_h(h_sample)
        prob_h_given_vk, h_sample_k = self.sample_h_given_v(prob_v_given_hk)

        for i in range(K-1):
            print(h_sample_k.shape)
            prob_v_given_hk,_ = self.sample_v_given_h(h_sample_k)
            prob_h_given_vk, h_sample_k = self.sample_h_given_v(prob_v_given_hk)
    

        #negative divergence
        negative_div = np.matmul(prob_v_given_hk.T, prob_h_given_vk)

        dweight = positive_div - negative_div
        dv_bias = visible - prob_v_given_hk
        dh_bias = prob_h_given_v - prob_h_given_vk

        self.weight += (self.lr/batch_size)*dweight
        self.v_bias += (self.lr/batch_size)*dv_bias
        self.h_bias += (self.lr/batch_size)*dh_bias
        cost = visible-prob_v_given_hk

        return cost


def sigmoid(input):
    return 1.0/(1 + np.exp(-input))

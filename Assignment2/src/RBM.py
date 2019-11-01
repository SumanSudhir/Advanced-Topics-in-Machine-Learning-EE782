import numpy as np
import math

class RBM():
    """Implementation of Restricted Boltzmann Machine(RBM)"""

    def __init__(self, n_visible, n_hidden, lr=0.001, l2_coeff=1e-4, momentum_coff=0.5):
        #Number of visible nodes
        self.n_visible = n_visible
        #Number of hidden nodes
        self.n_hidden = n_hidden
        #Weight initilization
        self.weight = np.random.randn(self.n_visible, self.n_hidden)/np.sqrt(n_visible)
        #Bias initilization
        self.v_bias = np.random.randn(1, self.n_visible)/np.sqrt(n_visible)
        self.h_bias = np.random.randn(1, self.n_hidden)/np.sqrt(n_visible)

        self.moment_weight = np.zeros((self.n_visible, self.n_hidden))
        self.moment_v_bias = np.zeros((1, self.n_visible))
        self.moment_h_bias = np.zeros((1, self.n_hidden))

        self.lr = lr
        self.l2_coeff = l2_coeff
        self.momentum_coff = momentum_coff

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

        # batch_size = np.shape(visible)[0]
        prob_h_given_v, h_sample = self.sample_h_given_v(visible)

        #positive divergence
        positive_div = np.matmul(visible.T, prob_h_given_v)

        prob_v_given_hk, v_sample_k = self.sample_v_given_h(h_sample)
        prob_h_given_vk, h_sample_k = self.sample_h_given_v(prob_v_given_hk)   #self.sample_h_given_v(v_sample_k)

        for i in range(K-1):
            # print(h_sample_k.shape)
            prob_v_given_hk, v_sample_k = self.sample_v_given_h(h_sample_k)
            prob_h_given_vk, h_sample_k = self.sample_h_given_v(prob_v_given_hk)    #self.sample_h_given_v(v_sample_k)


        #negative divergence
        negative_div = np.matmul(prob_v_given_hk.T, prob_h_given_vk)

        dweight = positive_div - negative_div
        dv_bias = visible - prob_v_given_hk
        dh_bias = prob_h_given_v - prob_h_given_vk

        prev_weight = self.weight.copy()
        #Momentum
        self.moment_weight *= self.momentum_coff
        self.moment_weight += dweight

        self.moment_v_bias *= self.momentum_coff
        self.moment_v_bias += dv_bias

        self.moment_h_bias *= self.momentum_coff
        self.moment_h_bias += dh_bias

        #Updating Weight and Bias
        self.weight += self.lr*self.moment_weight
        self.v_bias += self.lr*self.moment_v_bias
        self.h_bias += self.lr*self.moment_h_bias

        #L2 weight decay
        self.weight -= prev_weight*self.l2_coeff

        error = np.mean(np.abs(visible-prob_v_given_hk))
        #print(dh_bias)

        return error


def sigmoid(input):
    return 1.0/(1 + np.exp(-input))

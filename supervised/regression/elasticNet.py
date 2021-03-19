import numpy as np
from supervised.base.baseRegression import BaseRegression


class L1L2Regularization(object):
    def __init__(self, l1_ratio=.5, alpha=1):
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def __call__(self, weights):
        l1 = self.l1_ratio * np.sign(weights)
        l2 = (1 - self.l1_ratio) * weights
        return self.alpha * (l1 + l2)


class ElasticNet(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1, l1_ratio=.5):
        self.regularization = L1L2Regularization(l1_ratio=l1_ratio, alpha=alpha)
        super(ElasticNet, self).__init__(iterations=iterations, learning_rate=learning_rate)

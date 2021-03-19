
from supervised.base.baseRegression import BaseRegression
import numpy as np


class L1Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sign(weights)


class LassoRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1):
        self.regularization = L1Regularization(alpha)
        super(LassoRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

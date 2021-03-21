import numpy as np


class L1Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * np.sign(weights)


class L2Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * weights


class L1L2Regularization(object):
    def __init__(self, l1_ratio=.5, alpha=1):
        self.l1_ratio = l1_ratio
        self.alpha = alpha

    def __call__(self, weights):
        l1 = self.l1_ratio * np.sign(weights)
        l2 = (1 - self.l1_ratio) * weights
        return self.alpha * (l1 + l2)

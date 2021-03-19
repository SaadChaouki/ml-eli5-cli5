from supervised.base.baseRegression import BaseRegression


class L2Regularization(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, weights):
        return self.alpha * weights


class RidgeRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1):
        self.regularization = L2Regularization(alpha)
        super(RidgeRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

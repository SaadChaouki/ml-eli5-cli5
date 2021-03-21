from supervised.base.baseRegression import BaseRegression
from processing.regularization import L2Regularization


class RidgeRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1):
        self.regularization = L2Regularization(alpha)
        super(RidgeRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

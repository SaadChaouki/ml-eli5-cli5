from supervised.base.baseRegression import BaseRegression
from processing.regularization import L1Regularization


class LassoRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1):
        self.regularization = L1Regularization(alpha)
        super(LassoRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

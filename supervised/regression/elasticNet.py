from processing.regularization import L1L2Regularization
from supervised.base.baseRegression import BaseRegression


class ElasticNet(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, alpha=1, l1_ratio=.5):
        self.regularization = L1L2Regularization(l1_ratio=l1_ratio, alpha=alpha)
        super(ElasticNet, self).__init__(iterations=iterations, learning_rate=learning_rate)

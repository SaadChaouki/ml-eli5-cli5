from supervised.base.baseRegression import BaseRegression


class LinearRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01):
        self.regularization = lambda x: 0
        super(LinearRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        super(LinearRegression, self).fit(X, y)

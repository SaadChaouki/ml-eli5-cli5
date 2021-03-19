import numpy as np
from utils.activationFunctions import sigmoid
from supervised.base.baseRegression import BaseRegression


class LogisticRegression(BaseRegression):

    def __init__(self, learning_rate=.01, iterations=5000):
        self.regularization = lambda x: 0
        super(LogisticRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights))

    def predict_classes(self, X):
        return np.round(self.predict(X), 0)

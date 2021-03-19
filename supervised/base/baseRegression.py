import numpy as np
from utils.metrics import meanSquaredError


class BaseRegression(object):
    def __init__(self, iterations=5000, learning_rate=.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.error = []

    def compute_error(self, y, y_predicted):
        self.error.append(meanSquaredError(yTrue=y, yPred=y_predicted))

    def __compute_gradient(self, X, y, y_predicted):
        return np.dot(X.T, y_predicted - y) + self.regularization(self.weights)

    def fit(self, X, y):
        # Initialise weights and the bias
        self.weights = np.random.uniform(0, 0, (X.shape[1],))
        self.bias = 0

        for _ in range(self.iterations):
            y_predicted = self.predict(X)
            # self.compute_error(y, y_predicted)
            self.weights -= self.learning_rate * self.__compute_gradient(X, y, y_predicted)
            self.bias -= self.learning_rate * np.sum(y_predicted - y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


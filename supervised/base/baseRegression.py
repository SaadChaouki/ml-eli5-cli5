import numpy as np


class BaseRegression(object):
    def __init__(self, iterations=5000, learning_rate=.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = lambda x: 0
        self.bias = 0
        self.weights = []

    def __compute_gradient(self, X, y, y_predicted):
        return (np.dot(X.T, y_predicted - y) / len(y)) + self.regularization(self.weights)

    def fit(self, X, y):
        # Initialise weights and the bias
        self.weights = np.random.uniform(0, 0, (X.shape[1],))

        for _ in range(self.iterations):
            y_predicted = self.predict(X)
            self.weights -= self.learning_rate * self.__compute_gradient(X, y, y_predicted)
            self.bias -= self.learning_rate * np.mean(y_predicted - y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



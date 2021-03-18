import math
import numpy as np
from utils.activationFunctions import sigmoid


class LogisticRegression(object):

    def __init__(self, learning_rate=.1, iterations=5000):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, x, y):
        # Initialize the weights
        limit = 1 / math.sqrt(x.shape[1])
        self.weights = np.random.uniform(-limit, limit, (x.shape[1],))

        for i in range(self.iterations):
            yPredicted = self.predict_probabilities(x)
            self.weights -= self.learning_rate * (np.dot(x.T, yPredicted - y) / x.shape[0])

    def predict_probabilities(self, x):
        return sigmoid(np.dot(x, self.weights))

    def predict_classes(self, x):
        return np.round(self.predict_probabilities(x), 0)

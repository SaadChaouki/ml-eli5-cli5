import math

import numpy as np

from utils.activationFunctions import sigmoid


class LogisticRegression():

    def __init__(self, learningRate=.1):
        self.learningRate = learningRate

    def fit(self, x, y, iterations=5000):
        # Initialize the weights
        limit = 1 / math.sqrt(x.shape[1])
        self.weights = np.random.uniform(-limit, limit, (x.shape[1],))

        for i in range(iterations):
            yPredicted = self.predictProbabilities(x)
            self.weights -= self.learningRate * (np.dot(x.T, yPredicted - y) / x.shape[0])

    def predictProbabilities(self, x):
        return sigmoid(np.dot(x, self.weights))

    def predictClasses(self, x):
        return np.round(self.predictProbabilities(x), 0)
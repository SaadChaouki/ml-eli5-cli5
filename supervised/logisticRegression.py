import math

import numpy as np
from supervised.utils.activationFunctions import sigmoid
from supervised.utils.devutils import generateClassificationData
from supervised.utils.metrics import accuracy

class LogisticRegression():

    def __init__(self, learningRate = .1):
        self.learningRate = learningRate

    def fit(self, x, y, iterations = 1):
        # Initialize the weights
        limit = 1 / math.sqrt(x.shape[1])
        self.weights = np.random.uniform(-limit, limit, (x.shape[1],))

        for i in range(iterations):
            yPredicted = self.predictProbabilities(x)
            self.weights -= self.learningRate * (np.dot(x.T,  yPredicted - y)/x.shape[0])

    def predictProbabilities(self, x):
        return sigmoid(np.dot(x, self.weights))

    def predictClasses(self, x):
        return np.round(self.predictProbabilities(x), 0)

if __name__ == '__main__':
    # Generate data
    x, y = generateClassificationData(10)
    # Create LR
    lr = LogisticRegression()
    # fit
    lr.fit(x, y)
    # predict
    # yPredicted = lr.predictClasses(x)
    # yProbabilities = lr.predictProbabilities(x)
    # metrics
    # print(f'Accuracy: {accuracy(yPredicted, y)}')
    # print(f'Log Loss: {logLoss(yProbabilities, y)}')

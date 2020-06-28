from utils.devutils import *
from utils.metrics import logLoss, accuracy
from utils.activationFunctions import sigmoid
from sklearn.linear_model import LogisticRegression
import math
import numpy as np

class LogisticRegressionSelf():

    def __init__(self, learningRate = 0.01):
        self.weights = None
        self.learningRate = learningRate

    def fit(self, x, y, iterations = 2000):
        # Initialize the weights
        limit = 1 / math.sqrt(x.shape[1])
        self.weights = np.random.uniform(-limit, limit, (x.shape[1],))

        for i in range(iterations):
            yPredicted = self.predictProbabilities(x)
            computedError = logLoss(y, yPredicted)
            self.weights -= self.learningRate * (np.dot(x.T,  yPredicted - y)/x.shape[0])
            if i % 1000 == 0:
                print(f'Iteration {i} error {computedError} accuracy {accuracy(self.predictClasses(x), y)}')

    def predictProbabilities(self, x):
        return sigmoid(np.dot(x, self.weights))

    def predictClasses(self, x):
        return np.round(self.predictProbabilities(x), 0)


if __name__ == '__main__':
    generatedData = generateLRData(1000)
    x = np.array(generatedData[['var']].values)
    y = np.array(generatedData['target'].values)

    # Model
    lrSelf = LogisticRegressionSelf(learningRate = 0.001)
    lrSelf.fit(x, y, iterations = 20000)
    predictedClasses = lrSelf.predictClasses(x)
    prproba = lrSelf.predictProbabilities(x)
    print(accuracy(predictedClasses, y))

    # sklearn lr
    lr = LogisticRegression(max_iter = 2000)
    lr.fit(x, y)
    predictedsklearn = lr.predict(x)
    predictedproba = lr.predict_proba(x)
    print(accuracy(predictedsklearn, y))


    print('coefficients')
    print(lr.coef_)
    print(lrSelf.weights)

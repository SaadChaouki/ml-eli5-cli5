import numpy as np


def logLoss(yTrue, yPred):
    return - np.sum(((yTrue * np.log(yPred + 1e10)) + ((1 - yTrue) * np.log(1 - yPred + 1e10)))) / len(yTrue)


def accuracy(yTrue, yPred):
    return np.sum(yTrue == yPred) / yTrue.size


def meanSquaredError(yTrue, yPred):
    return np.sum((yTrue - yPred) ** 2) / yTrue.size


def meanAbsoluteError(yTrue, yPred):
    return np.sum(np.abs(yTrue - yPred)) / yTrue.size


def precision(yTrue, yPred):
    # todo: create precision function
    return None


def recall(yTrue, yPred):
    # todo: create recall function
    return None


if __name__ == '__main__':
    testingPredictions = np.array([2.5, 0.0, 2, 8])
    testingTruth = np.array([3, -0.5, 2, 7])

    print(meanAbsoluteError(testingTruth, testingPredictions))

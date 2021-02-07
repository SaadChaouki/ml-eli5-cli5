from utils.devutils import generateClassificationData
import numpy as np

def covariance(a, b):
    cov = sum((a - a.mean()) * (b - b.mean())) / (len(a) - 1)
    return cov


def covarianceMatrix(x):
    covMatrix = (1 / (x.shape[0] - 1)) * (x - x.mean(axis=0)).T.dot(x - x.mean(axis=0))
    return covMatrix


def cFactor(n):
    return (2 * (np.log(n - 1    ) + 0.5772156649)) - ((2 * (n - 1)) / n)

if __name__ == '__main__':
    x, y = generateClassificationData(5)
    m = covarianceMatrix(x)

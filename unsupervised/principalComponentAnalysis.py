
from utils.devutils import generateClassificationData
from utils.functions import covarianceMatrix
import numpy as np
import matplotlib.pyplot as plt

class PCA():
    def __init__(self, nComponents):
        self.nComponents = nComponents

    def transform(self, x):
        covMatrix = covarianceMatrix(x)
        eigenValues, eigenVectors = np.linalg.eigh(covMatrix)
        sortedIdx = np.argsort(eigenValues)[::-1]
        eigenVectors = eigenVectors[:, sortedIdx][:, :self.nComponents]
        return x.dot(eigenVectors)

if __name__ == '__main__':
    pca = PCA(2)
    x, y = generateClassificationData(1000)
    t = pca.transform(x)




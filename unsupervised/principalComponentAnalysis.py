
from utils.devutils import generateClassificationData
from utils.functions import covarianceMatrix
import numpy as np


class PCA():
    def __init__(self, nComponents):
        self.nComponents = nComponents

    def transform(self, x):
        covMatrix = covarianceMatrix(x)
        eigenValues, eigenVectors = np.linalg.eigh(covMatrix)
        print(eigenValues)

if __name__ == '__main__':
    pca = PCA(2)
    x, y = generateClassificationData(100)
    x[:, 0] = x[:, 1]
    pca.transform(x)


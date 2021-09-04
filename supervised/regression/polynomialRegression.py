from processing.polynomialTransformer import PolynomialTransformation
from supervised.base.baseRegression import BaseRegression

import numpy as np


class PolynomialRegression(BaseRegression):
    def __init__(self, iterations: int = 5000, learning_rate: float = .01, degree: int = 2):
        super(PolynomialRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)
        self.poly_transformed = PolynomialTransformation(degree=degree, include_bias=False)

    def fit(self, X: np.array, y: np.array):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).fit(X, y)

    def transform_predict(self, X: np.array):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).predict(X)

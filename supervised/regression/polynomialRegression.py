import itertools
from itertools import combinations_with_replacement

import numpy as np

from supervised.base.baseRegression import BaseRegression


class PolynomialTransformation(object):
    def __init__(self, degree: int = 2, include_bias: bool = False):
        self.degree = degree
        self.include_bias = include_bias

    def transform(self, X: np.array):
        self.n_samples, self.n_features = X.shape
        self.__create_combinations()

        poly_features = X.copy()
        for combination in self.combinations:
            poly_features = np.append(poly_features, np.prod(X[:, combination], axis=1, keepdims=True), axis=1)

        if self.include_bias:
            poly_features = np.append(np.ones(shape=(self.n_samples, 1)), poly_features, axis=1)
        return poly_features

    def __create_combinations(self):
        combinations = [combinations_with_replacement(range(self.n_features), i) for i in range(2, self.degree + 1)]
        self.combinations = list(itertools.chain.from_iterable(combinations))


class PolynomialRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, degree=2):
        self.regularization = lambda x: 0
        self.poly_transformed = PolynomialTransformation(degree=degree, include_bias=False)
        super(PolynomialRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).fit(X, y)

    def transform_predict(self, X):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).predict(X)

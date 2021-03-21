import numpy as np
import itertools as it


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
        combinations = [it.combinations_with_replacement(range(self.n_features), i) for i in range(2, self.degree + 1)]
        self.combinations = list(it.chain.from_iterable(combinations))

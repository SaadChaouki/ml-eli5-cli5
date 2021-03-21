from processing.polynomialTransformer import PolynomialTransformation
from supervised.base.baseRegression import BaseRegression


class PolynomialRegression(BaseRegression):
    def __init__(self, iterations=5000, learning_rate=.01, degree=2):
        self.poly_transformed = PolynomialTransformation(degree=degree, include_bias=False)
        super(PolynomialRegression, self).__init__(iterations=iterations, learning_rate=learning_rate)

    def fit(self, X, y):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).fit(X, y)

    def transform_predict(self, X):
        X = self.poly_transformed.transform(X)
        return super(PolynomialRegression, self).predict(X)

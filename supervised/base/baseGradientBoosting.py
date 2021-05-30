import numpy as np
from supervised.regression.decisionTreeRegressor import DecisionTreeRegressor


class BaseGradientBoosting(object):
    def __init__(self, max_depth=2, num_estimators=100, minimum_sample_leaf=10, learning_rate=.1):
        self.max_depth = max_depth
        self.num_estimators = num_estimators
        self.minimum_sample_leaf = minimum_sample_leaf
        self.learning_rate = learning_rate
        self.models = [DecisionTreeRegressor(
            max_depth=self.max_depth,
            minimum_sample_leaf=self.minimum_sample_leaf
        ) for _ in range(self.num_estimators)]
        self.loss = None
        self.transformation = None

    def fit(self, X, y):
        # Starting with the average of y
        y_predicted = np.full(len(y), np.mean(y))

        # Fitting the models
        for model in self.models:
            gradient = self.loss.gradient(y, y_predicted)
            model.fit(X, gradient)
            gradient_prediction = model.predict(X)
            y_predicted -= self.learning_rate * np.array(gradient_prediction)

    def predict(self, X):
        y_pred = np.array([])
        # Make predictions
        for model in self.models:
            update = model.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        return self.transformation(y_pred)
from supervised.base.baseDecisionTree import BaseDecisionTree
import numpy as np


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, max_depth=2, minimum_sample_leaf=10):
        super(DecisionTreeClassifier, self).__init__(max_depth=max_depth, minimum_sample_leaf=minimum_sample_leaf)
        self.gain_function = self.__compute_gain
        self.prediction_function = self.__get_majority_vote

    @staticmethod
    def __compute_gain(y):
        countClasses = [np.sum(y == c) for c in np.unique(y)]
        return 1.0 - np.sum([(countClass / y.size) ** 2 for countClass in countClasses])

    @staticmethod
    def __get_majority_vote(y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from supervised.base.baseRandomForest import BaseRandomForest
import numpy as np


class RandomForestClassifier(BaseRandomForest):
    def __init__(self, num_estimators=100, max_depth=2, max_features=None, min_sample_leaf=10):
        super(RandomForestClassifier, self).__init__(num_estimators=num_estimators, max_depth=max_depth,
                                                     max_features=max_features, min_sample_leaf=min_sample_leaf)
        self.prediction_combination = self.__majority_vote
        self.tree_model = DecisionTreeClassifier

    @staticmethod
    def __majority_vote(predictions):
        classes, counts = np.unique(predictions, return_counts=True)
        return classes[np.argmax(counts)]

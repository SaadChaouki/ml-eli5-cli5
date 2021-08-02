from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from supervised.classification.naiveBayes import NaiveBayesClassifier
from supervised.classification.randomForestClassifier import RandomForestClassifier
from supervised.classification.gradientBoostingClassifier import GradientBoostingClassifier
from supervised.classification.logisticRegression import LogisticRegression
from supervised.classification.KNearestNeighbor import KNN
from supervised.classification.adaBoost import AdaBoost

from sklearn.datasets import make_classification, make_moons, make_circles
import matplotlib.pyplot as plt
import matplotlib


matplotlib.use("TkAgg")

if __name__ == '__main__':
    # Parameters
    grid_detail = 1.
    data_size = 100

    # Create the dataset
    datasets = {
        'linear': None,
        'moons': None,
        'circles': None
    }

    # Create the models
    models = {
        'K-Nearest Neighbor': KNN(k=5)
    }

    # Start the pyplot grid
    figure = plt.figure(figsize=(27, 9))


from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from supervised.classification.naiveBayes import NaiveBayesClassifier
from supervised.classification.randomForestClassifier import RandomForestClassifier
from supervised.classification.gradientBoostingClassifier import GradientBoostingClassifier
from supervised.classification.logisticRegression import LogisticRegression
from supervised.classification.KNearestNeighbor import KNN
from supervised.classification.adaBoost import AdaBoost

from color_palette import two_colors, two_colors_map

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
        'linear': make_classification(n_features=2, n_redundant=0, n_informative=2),
        'moons': make_moons(noise=0.3, random_state=0),
        'circles': make_circles(noise=0.2, factor=0.5, random_state=1)
    }

    # Create the models
    models = {
        'K-Nearest Neighbor': KNN(k=5)
    }

    # Start the pyplot grid
    figure, axes = plt.subplots(len(datasets), len(models) + 1, figsize=(27, 9))

    # Plot the input data
    for i, data_name in enumerate(datasets):
        ax = axes[i][0]
        if i == 0:
            ax.set_title('Input Data')
        X, y = datasets[data_name]
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=two_colors_map, edgecolors='k')
        ax.set_xticks(())
        ax.set_yticks(())


    plt.show()

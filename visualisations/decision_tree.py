from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from visualisations.color_palette import two_colors, two_colors_map
from utils.metrics import accuracy

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import argparse


matplotlib.use("TkAgg")

if __name__ == '__main__':

    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a custom Decision Tree Classifier model in training.')
    parser.add_argument('--max_depth', type=int, help='Maximum depth of the tree.', default=15)
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=1000)
    parser.add_argument('--test_size', type=float, help='Test set size.', default=.2)
    args = parser.parse_args()

    # Parameters
    max_depth = args.max_depth

    X, y = make_classification(n_samples=args.n_samples, n_features=2, n_informative=2, n_redundant=0,
                               random_state=args.random_state)

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Model
    classifiers = [DecisionTreeClassifier(max_depth=i + 1, minimum_sample_leaf=1) for i in range(max_depth)]

    # Create decision boundary data
    h = 2
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    area_data = np.c_[xx.ravel(), yy.ravel()]

    # Fitting the models
    predictions_test = []
    predictions_train = []
    area_pred = []
    for clf in classifiers:
        clf.fit(X_train, y_train)
        predictions_test.append(clf.predict(X_test))
        predictions_train.append(clf.predict(X_train))
        area = np.array(clf.predict(area_data))
        area_pred.append(area.reshape(xx.shape))

    # Create the update function for the graph
    def update(i):
        plt.clf()
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        fig.suptitle('Decision Tree Classifier', fontsize=20)
        plt.title(f'Decision Tree Depth: {i + 1} - '
                  f'Accuracy Test: {round(100 * accuracy(y_test, predictions_test[i]), 2)}% '
                  f'Accuracy Train: {round(100 * accuracy(y_train, predictions_train[i]), 2)}% '
                  )
        plt.scatter(X[:, 0], X[:, 1], c=[two_colors[k] for k in y])
        plt.imshow(area_pred[i], interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=two_colors_map, aspect='auto', origin='lower', alpha=.4)

    # Plotting and saving the gif
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    animation = FuncAnimation(fig, update, frames=max_depth, interval=800, repeat=False)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
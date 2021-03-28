from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from matplotlib.animation import FuncAnimation, PillowWriter
from utils.metrics import accuracy
import numpy as np
from visualisations.color_palette import two_colors, two_colors_map

matplotlib.use("TkAgg")

if __name__ == '__main__':
    max_depth = 20

    X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=30)

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Model
    classifiers = [DecisionTreeClassifier(max_depth=i + 1, minimum_sample_leaf=1) for i in range(max_depth)]

    # Create decision boundary data
    h = .05
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
    animation.save('animations/decision_tree.gif', writer=PillowWriter(fps=60))
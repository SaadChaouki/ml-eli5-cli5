from supervised.regression.decisionTreeRegressor import DecisionTreeRegressor
from utils.metrics import meanSquaredError as mse
from visualisations.color_palette import two_colors

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.model_selection import train_test_split

import argparse

matplotlib.use("TkAgg")


def update(i):
    y_pred = np.array([x for _, x in sorted(zip(X_train, train_predictions[i]))])
    plt.title(f'Tree Depth: {i + 1} | '
              f'MSE Train: {round(mse(y_train, train_predictions[i]), 2)}'
              f' | MSE Test: {round(mse(y_test, test_predictions[i]), 2)}')
    line.set_ydata(y_pred)


if __name__ == '__main__':

    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a custom Decision Tree Regressor model in training.')
    parser.add_argument('--max_depth', type=int, help='Maximum depth of the tree.', default=15)
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=500)
    parser.add_argument('--test_size', type=float, help='Test set size.', default=.2)
    args = parser.parse_args()

    # Getting the arguments
    np.random.seed(args.random_state)
    max_depth = args.max_depth
    n_samples = args.n_samples

    # Data generation
    X = np.random.normal(0, 1, n_samples)
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-2, 2, n_samples)

    # Reshaping
    X = np.atleast_2d(X).reshape(-1, 1)

    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Model and Predictions
    classifiers = [DecisionTreeRegressor(max_depth=i + 1) for i in range(max_depth)]

    # Fitting and predicting
    train_predictions = []
    test_predictions = []
    for clf in classifiers:
        clf.fit(X_train, y_train)
        train_predictions.append(clf.predict(X_train))
        test_predictions.append(clf.predict(X_test))

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    fig.suptitle('Regression Tree', fontsize=20)

    # Original Data - train and test
    ax.scatter(X_train, y_train, color=two_colors[0], label='Train Data')
    ax.scatter(X_test, y_test, color=two_colors[1], label='Test Data')

    # Sort and plot initial line
    y_pred = np.array([x for _, x in sorted(zip(X_train, train_predictions[0]))])
    X_train_sorted = np.array(sorted(X_train))
    line, = ax.plot(X_train_sorted, y_pred, color='black', linewidth=2, label="Prediction")

    # Labels and legend
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend(loc='lower right')

    # Animation
    animation = FuncAnimation(fig, update, frames=max_depth, interval=1000, repeat=False)

    # Showing the plot
    plt.show()

from supervised.regression.polynomialRegression import PolynomialRegression
from visualisations.color_palette import two_colors
from deep_learning.loss import MSELoss

from sklearn.model_selection import train_test_split

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
import argparse

matplotlib.use("TkAgg")


def update(i):
    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[i]))])
    plt.title(f'Iteration: {i + 1} | MSE: {round(MSELoss()(y_train, model.error[i]), 2)}')
    line.set_ydata(y_pred)


if __name__ == '__main__':

    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a customer Polynomial Regression model in training.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=100)
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=500)
    parser.add_argument('--test_size', type=float, help='Test set size.', default=.2)
    parser.add_argument('--lr', type=float, help='Learning Rate.', default=.01)
    parser.add_argument('--degree', type=int, help='Degrees.', default=3)
    args = parser.parse_args()

    # Setting up parameters.
    np.random.seed(args.random_state)
    max_iterations = args.max_iter
    n_features = args.n_samples

    # Data generation
    X = np.random.normal(0, 1, n_features)
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-2, 2, n_features)

    # Reshaping
    X = np.atleast_2d(X).reshape(-1, 1)

    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size,
                                                        random_state=args.random_state, shuffle=True)

    # Model and Predictions
    model = PolynomialRegression(learning_rate=args.lr, iterations=max_iterations, degree=args.degree)
    model.fit(X_train, y_train)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    fig.suptitle('Polynomial Regression', fontsize=20)

    # Plot original training and testing data.
    ax.scatter(X_train, y_train, color=two_colors[0], label='Train Data')
    ax.scatter(X_test, y_test, color=two_colors[1], label='Test Data')

    # Plot first iteration line
    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[0]))])
    X_train_sorted = np.array(sorted(X_train))
    line, = ax.plot(X_train_sorted, y_pred, color='black', linewidth=2, label="Prediction")

    # Labels and legend.
    plt.legend(loc='lower right')
    plt.xlabel('Feature')
    plt.ylabel('Target')

    # Animation
    animation = FuncAnimation(fig, update, frames=max_iterations, interval=1, repeat=False)

    # Show plot
    plt.show()
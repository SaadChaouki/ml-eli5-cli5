from supervised.regression.linearRegression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.metrics import meanSquaredError as mse
from matplotlib.animation import FuncAnimation
import matplotlib
from visualisations.color_palette import two_colors
import numpy as np
import argparse

matplotlib.use("TkAgg")


def update(i):
    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[i]))])
    plt.title(f'Iteration: {i + 1} | MSE: {round(mse(y_train, model.error[i]), 2)}')
    line.set_ydata(y_pred)


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a customer Linear Regression model in training.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=100)
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=500)
    parser.add_argument('--test_size', type=float, help='Test set size.', default=.2)
    parser.add_argument('--lr', type=float, help='Learning Rate.', default=.1)
    args = parser.parse_args()
    max_iterations = args.max_iter

    # Generate regression data
    X, y = make_regression(n_features=1, n_samples=args.n_samples, n_informative=1, noise=30,
                           random_state=args.random_state, bias=500, tail_strength=1)

    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    # Model and Predictions
    model = LinearRegression(learning_rate=args.lr, iterations=max_iterations)
    model.fit(X_train, y_train)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    fig.suptitle('Linear Regression', fontsize=20)

    # Original Data
    ax.scatter(X_train, y_train, color=two_colors[0], label='Train Data')
    ax.scatter(X_test, y_test, color=two_colors[1], label='Test Data')
    plt.xlabel('Feature')
    plt.ylabel('Target')

    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[0]))])
    X_train_sorted = np.array(sorted(X_train))

    line, = ax.plot(X_train_sorted, y_pred, color='black', linewidth=2, label="Prediction")

    plt.legend(loc='lower right')

    animation = FuncAnimation(fig, update, frames=max_iterations, interval=1, repeat=False)
    plt.show()

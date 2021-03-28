from supervised.regression.linearRegression import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.metrics import meanSquaredError as mse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
from visualisations.color_palette import two_colors
import numpy as np

matplotlib.use("TkAgg")


def update(i):
    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[i]))])
    plt.title(f'Iteration: {i + 1} | MSE: {round(mse(y_train, model.error[i]), 2)}')
    line.set_ydata(y_pred)


if __name__ == '__main__':
    max_iterations = 100

    # Generate regression data
    X, y = make_regression(n_features=1, n_samples=500, n_informative=1, noise=30,
                           random_state=42, bias=500, tail_strength=1)

    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Model and Predictions
    model = LinearRegression(learning_rate=.1, iterations=max_iterations)
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
    animation.save('animations/linear_regression.gif', writer=PillowWriter(fps=60))

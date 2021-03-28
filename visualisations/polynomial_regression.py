from supervised.regression.polynomialRegression import PolynomialRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from utils.metrics import meanSquaredError as mse
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
from visualisations.color_palette import two_colors

matplotlib.use("TkAgg")


def update(i):
    y_pred = np.array([x for _, x in sorted(zip(X_train, model.error[i]))])
    plt.title(f'Iteration: {i + 1} | MSE: {round(mse(y_train, model.error[i]), 2)}')
    line.set_ydata(y_pred)


if __name__ == '__main__':
    np.random.seed(42)
    max_iterations = 150

    n_features = 500

    # Data generation
    X = np.random.normal(0, 1, n_features)
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-2, 2, n_features)

    # Reshaping
    X = np.atleast_2d(X).reshape(-1, 1)

    # Train - Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, shuffle=True)

    # Model and Predictions
    model = PolynomialRegression(learning_rate=.01, iterations=max_iterations, degree=3)
    model.fit(X_train, y_train)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    fig.suptitle('Polynomial Regression', fontsize=20)

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
    animation.save('animations/polynomial_regression.gif', writer=PillowWriter(fps=60))

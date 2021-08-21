import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from color_palette import two_colors
from supervised.regression.decisionTreeRegressor import DecisionTreeRegressor
from supervised.regression.elasticNet import ElasticNet
from supervised.regression.gradientBoostingRegressor import GradientBoostingRegressor
from supervised.regression.lassoRegression import LassoRegression
from supervised.regression.linearRegression import LinearRegression
from supervised.regression.polynomialRegression import PolynomialRegression
from supervised.regression.randomForestRegressor import RandomForestRegressor
from supervised.regression.ridgeRegression import RidgeRegression

matplotlib.use("TkAgg")


def generate_polynomial(n_samples: int = 100):
    X = np.random.normal(0, 1, n_samples)
    y = X - 2 * (X ** 2) + 0.5 * (X ** 3) + np.random.normal(-2, 2, n_samples)
    X = np.atleast_2d(X).reshape(-1, 1)
    return X, y


if __name__ == '__main__':
    # Generate the different datasets
    datasets = {
        'linear': make_regression(n_features=1, n_samples=100, n_informative=1, noise=30, random_state=42, bias=500,
                                  tail_strength=1),
        'polynomial': generate_polynomial(),
    }

    # Create the models
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=10),
        'Polynomial Regression': PolynomialRegression(degree=3)
    }

    # Start the pyplot grid
    figure, axes = plt.subplots(len(datasets), len(models) + 1, figsize=(27, 9))
    figure.suptitle('Machine Learning - ELI5 - CLI5 - Regression', size=20)

    # Plot the input data
    for i, data_name in enumerate(datasets):
        # Set the title
        if i == 0:
            axes[i][0].set_title('Input Data')

        # Extract the data from the dictionary
        X, y = datasets[data_name]

        # Plot the data
        axes[i][0].scatter(X, y, edgecolors='k', c=two_colors[1])

        # Remove the axes
        axes[i][0].set_xticks(())
        axes[i][0].set_yticks(())

    # Process the models
    for i, model_name in enumerate(models):
        for j, data_name in enumerate(datasets):
            X, y = datasets[data_name]

            # Set the title
            if j == 0:
                axes[j][i + 1].set_title(model_name)

            # Plot the data
            axes[j][i + 1].scatter(X, y, edgecolors='k', c=two_colors[1])

            # Train the model
            models[model_name].fit(X, y)

            # Predict
            if callable(getattr(models[model_name], "transform_predict", None)):
                y_pred = models[model_name].transform_predict(X)
                print(f'Calling transform predict for the model {model_name}.')
            else:
                y_pred = models[model_name].predict(X)

            # Organising the data to plot a line
            y_pred = np.array([x for _, x in sorted(zip(X, y_pred))])
            X_sorted = np.array(sorted(X))

            # Plotting the results
            axes[j][i + 1].plot(X_sorted, y_pred, color=two_colors[0], linewidth=2)

            # Remove the axes
            axes[j][i + 1].set_xticks(())
            axes[j][i + 1].set_yticks(())

    # Save and show the plot
    plt.show()
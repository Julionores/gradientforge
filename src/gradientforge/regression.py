"""Moteur de regression lineaire (simple, polynomiale, multivariee) en NumPy pur.

Aucune dependance a scikit-learn n'est utilisee pour l'entrainement : ce module
implemente explicitement la matrice de design, la fonction de cout MSE, son
gradient analytique, et la descente de gradient batch qui les relie.
"""

import numpy as np


def make_design_matrix(*columns):
    """Empile des colonnes de features et ajoute une colonne de biais (1)."""
    X = np.column_stack(columns)
    bias = np.ones((X.shape[0], 1))
    return np.hstack([X, bias])


def standardize(x):
    """Centre-reduit un tableau (moyenne 0, ecart-type 1)."""
    return (x - x.mean()) / x.std()


def linear_model(X, theta):
    return X @ theta


def mse_cost(X, y, theta):
    m = len(y)
    return (1 / (2 * m)) * np.sum((linear_model(X, theta) - y) ** 2)


def mse_grad(X, y, theta):
    m = len(y)
    return (1 / m) * X.T @ (linear_model(X, theta) - y)


def gradient_descent(
    X, y, theta, learning_rate, n_iterations, cost_fn=mse_cost, grad_fn=mse_grad
):
    """Descente de gradient batch generique : fonctionne pour la regression
    (mse_cost/mse_grad, par defaut) comme pour la classification (voir
    classification.py), tant que cost_fn/grad_fn partagent la meme signature.
    """
    cost_history = np.zeros(n_iterations)
    for i in range(n_iterations):
        theta = theta - learning_rate * grad_fn(X, y, theta)
        cost_history[i] = cost_fn(X, y, theta)
    return theta, cost_history


def r_squared(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot

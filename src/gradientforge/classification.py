"""Moteur de regression logistique (classification binaire) en NumPy pur.

Reutilise la meme matrice de design et la meme mecanique de descente de
gradient batch que regression.py : seule la definition du modele (sigmoide
au lieu d'identite) et de la fonction de cout (log-loss au lieu de MSE)
changent.
"""

import numpy as np


def sigmoid(z):
    """Sigmoide numeriquement stable : evite l'overflow de exp() pour les z tres negatifs
    en distinguant le calcul selon le signe de z (les deux formes sont mathematiquement
    equivalentes : 1/(1+e^-z) = e^z/(1+e^z))."""
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)
    positive = z >= 0
    result[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    result[~positive] = exp_z / (1 + exp_z)
    return result


def logistic_model(X, theta):
    return sigmoid(X @ theta)


def log_loss(X, y, theta, eps=1e-15):
    m = len(y)
    A = logistic_model(X, theta)
    A = np.clip(A, eps, 1 - eps)
    return -(1 / m) * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))


def log_loss_grad(X, y, theta):
    m = len(y)
    return (1 / m) * X.T @ (logistic_model(X, theta) - y)


def predict_class(X, theta, threshold=0.5):
    return (logistic_model(X, theta) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))

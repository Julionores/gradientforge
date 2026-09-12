"""Demo : classer un flux reseau comme normal ou a risque, avec le moteur de
classification 'from scratch' de gradientforge."""

import numpy as np

from gradientforge import (
    make_design_matrix,
    standardize,
    gradient_descent,
    log_loss,
    log_loss_grad,
    predict_class,
    accuracy,
)

rng = np.random.default_rng(0)
n_per_class = 100

debit_normal = rng.normal(loc=50, scale=8, size=n_per_class)
latence_normal = rng.normal(loc=20, scale=5, size=n_per_class)
debit_risque = rng.normal(loc=15, scale=8, size=n_per_class)
latence_risque = rng.normal(loc=80, scale=10, size=n_per_class)

debit = np.concatenate([debit_normal, debit_risque])
latence = np.concatenate([latence_normal, latence_risque])
label = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).reshape(-1, 1)

X = make_design_matrix(standardize(debit), standardize(latence))

indices = rng.permutation(len(label))
split = int(0.8 * len(label))
X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = label[indices[:split]], label[indices[split:]]

theta_final, loss_history = gradient_descent(
    X_train,
    y_train,
    rng.standard_normal((3, 1)),
    learning_rate=0.1,
    n_iterations=3000,
    cost_fn=log_loss,
    grad_fn=log_loss_grad,
)

pred_test = predict_class(X_test, theta_final)
model_accuracy = accuracy(y_test, pred_test)

classe_majoritaire = round(y_train.mean())
baseline_pred = np.full_like(y_test, classe_majoritaire)
baseline_accuracy = accuracy(y_test, baseline_pred)

print("=== BoundaryScope : classification de risque reseau ===")
print("perte initiale (log-loss):", loss_history[0])
print("perte finale (log-loss):", loss_history[-1])
print("accuracy test (modele):", model_accuracy)
print("accuracy test (baseline classe majoritaire):", baseline_accuracy)
print("theta appris [debit, latence, biais]:", theta_final.ravel())

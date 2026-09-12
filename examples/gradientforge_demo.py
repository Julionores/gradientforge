"""Demo : prevoir la consommation electrique d'un data center a partir de sa
telemetrie (charge CPU, RAM, trafic reseau), avec le moteur de regression
'from scratch' de gradientforge."""

import numpy as np

from gradientforge import (
    make_design_matrix,
    standardize,
    linear_model,
    gradient_descent,
    r_squared,
)

rng = np.random.default_rng(42)
n_samples = 100

# --- 1. Regression simple : charge CPU -> consommation ---
charge_cpu = rng.uniform(10, 95, n_samples)
bruit = rng.normal(0, 15, n_samples)
consommation = 8 * charge_cpu + 120 + bruit

charge_cpu_std = standardize(charge_cpu)
X_simple = make_design_matrix(charge_cpu_std)
y = consommation.reshape(-1, 1)

theta_final, cost_history = gradient_descent(
    X_simple, y, rng.standard_normal((2, 1)), learning_rate=0.05, n_iterations=2000
)
print("=== Regression simple (charge CPU -> consommation) ===")
print("theta appris (pente, biais):", theta_final.ravel())
print("cout initial:", cost_history[0])
print("cout final:", cost_history[-1])
print("R2:", r_squared(y, linear_model(X_simple, theta_final)))

# --- 2. Regression polynomiale : effet de throttling thermique ---
consommation_nl = 8 * charge_cpu + 0.05 * charge_cpu**2 + 120 + bruit
X_poly = make_design_matrix(charge_cpu_std, standardize(charge_cpu**2))
y_nl = consommation_nl.reshape(-1, 1)

theta_poly, _ = gradient_descent(
    X_poly, y_nl, rng.standard_normal((3, 1)), learning_rate=0.05, n_iterations=3000
)
theta_lin, _ = gradient_descent(
    X_simple, y_nl, rng.standard_normal((2, 1)), learning_rate=0.05, n_iterations=2000
)
print("\n=== Regression polynomiale vs lineaire (donnees non lineaires) ===")
print("R2 polynomial:", r_squared(y_nl, linear_model(X_poly, theta_poly)))
print("R2 lineaire (sous-ajuste):", r_squared(y_nl, linear_model(X_simple, theta_lin)))

# --- 3. Regression multivariee ---
ram_pct = rng.uniform(20, 90, n_samples)
trafic_reseau = rng.uniform(5, 500, n_samples)
consommation_multi = (
    6 * charge_cpu
    + 3 * ram_pct
    + 0.2 * trafic_reseau
    + 80
    + rng.normal(0, 20, n_samples)
)
X_multi = make_design_matrix(
    standardize(charge_cpu), standardize(ram_pct), standardize(trafic_reseau)
)
y_multi = consommation_multi.reshape(-1, 1)

theta_multi, _ = gradient_descent(
    X_multi, y_multi, rng.standard_normal((4, 1)), learning_rate=0.05, n_iterations=5000
)
print("\n=== Regression multivariee (CPU + RAM + reseau) ===")
print("theta appris [cpu, ram, reseau, biais]:", theta_multi.ravel())
print("R2:", r_squared(y_multi, linear_model(X_multi, theta_multi)))

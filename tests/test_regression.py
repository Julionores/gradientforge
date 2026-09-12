import numpy as np
import pytest

from gradientforge import (
    make_design_matrix,
    standardize,
    linear_model,
    gradient_descent,
    r_squared,
)


@pytest.fixture
def energy_data():
    """Consommation electrique (kWh) d'un data center en fonction de la charge CPU (%)."""
    rng = np.random.default_rng(42)
    n_samples = 100
    charge_cpu = rng.uniform(10, 95, n_samples)
    bruit = rng.normal(0, 15, n_samples)
    consommation = 8 * charge_cpu + 120 + bruit
    return charge_cpu, consommation


def test_make_design_matrix_adds_bias_column():
    X = make_design_matrix(np.array([1.0, 2.0, 3.0]))
    assert X.shape == (3, 2)
    assert np.all(X[:, -1] == 1.0)


def test_make_design_matrix_stacks_multiple_columns():
    X = make_design_matrix(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    assert X.shape == (2, 3)
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0])
    np.testing.assert_array_equal(X[:, 1], [10.0, 20.0])


def test_standardize_produces_zero_mean_unit_std():
    x = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    x_std = standardize(x)
    assert np.isclose(x_std.mean(), 0.0, atol=1e-10)
    assert np.isclose(x_std.std(), 1.0, atol=1e-10)


def test_unscaled_features_with_high_learning_rate_diverge(energy_data):
    """Documente le cas reel de divergence rencontre en preparant ce projet :
    sans standardisation, un learning_rate pourtant modeste fait exploser le cout."""
    charge_cpu, consommation = energy_data
    X = make_design_matrix(charge_cpu)  # PAS de standardisation
    y = consommation.reshape(-1, 1)
    theta_init = np.random.default_rng(1).standard_normal((2, 1))

    with np.errstate(over="ignore", invalid="ignore"):
        _, cost_history = gradient_descent(
            X, y, theta_init, learning_rate=0.0007, n_iterations=200
        )

    assert (
        cost_history[-1] > cost_history[0]
    ), "le cout doit exploser avec des features non standardisees"


def test_standardized_features_converge_and_fit_well(energy_data):
    charge_cpu, consommation = energy_data
    charge_cpu_std = standardize(charge_cpu)
    X = make_design_matrix(charge_cpu_std)
    y = consommation.reshape(-1, 1)
    theta_init = np.random.default_rng(1).standard_normal((2, 1))

    theta_final, cost_history = gradient_descent(
        X, y, theta_init, learning_rate=0.05, n_iterations=2000
    )
    predictions = linear_model(X, theta_final)

    assert cost_history[-1] < cost_history[0]
    assert r_squared(y, predictions) > 0.95


def test_polynomial_beats_linear_on_nonlinear_data(energy_data):
    charge_cpu, _ = energy_data
    rng = np.random.default_rng(42)
    bruit = rng.normal(0, 15, len(charge_cpu))
    consommation_nl = 8 * charge_cpu + 0.05 * charge_cpu**2 + 120 + bruit
    y_nl = consommation_nl.reshape(-1, 1)

    charge_cpu_std = standardize(charge_cpu)
    charge_cpu_sq_std = standardize(charge_cpu**2)

    X_poly = make_design_matrix(charge_cpu_std, charge_cpu_sq_std)
    theta_poly, _ = gradient_descent(
        X_poly,
        y_nl,
        np.random.default_rng(2).standard_normal((3, 1)),
        learning_rate=0.05,
        n_iterations=3000,
    )
    r2_poly = r_squared(y_nl, linear_model(X_poly, theta_poly))

    X_lin = make_design_matrix(charge_cpu_std)
    theta_lin, _ = gradient_descent(
        X_lin,
        y_nl,
        np.random.default_rng(2).standard_normal((2, 1)),
        learning_rate=0.05,
        n_iterations=2000,
    )
    r2_lin = r_squared(y_nl, linear_model(X_lin, theta_lin))

    assert r2_poly > r2_lin


def test_multivariate_regression_achieves_good_fit():
    rng = np.random.default_rng(42)
    n_samples = 100
    charge_cpu = rng.uniform(10, 95, n_samples)
    ram_pct = rng.uniform(20, 90, n_samples)
    trafic_reseau = rng.uniform(5, 500, n_samples)
    consommation = (
        6 * charge_cpu
        + 3 * ram_pct
        + 0.2 * trafic_reseau
        + 80
        + rng.normal(0, 20, n_samples)
    )

    X = make_design_matrix(
        standardize(charge_cpu), standardize(ram_pct), standardize(trafic_reseau)
    )
    y = consommation.reshape(-1, 1)

    theta_final, _ = gradient_descent(
        X,
        y,
        np.random.default_rng(3).standard_normal((4, 1)),
        learning_rate=0.05,
        n_iterations=5000,
    )
    predictions = linear_model(X, theta_final)
    assert r_squared(y, predictions) > 0.7


def test_matches_sklearn_linear_regression(energy_data):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LinearRegression

    charge_cpu, consommation = energy_data
    charge_cpu_std = standardize(charge_cpu)
    X = make_design_matrix(charge_cpu_std)
    y = consommation.reshape(-1, 1)

    theta_final, _ = gradient_descent(
        X,
        y,
        np.random.default_rng(1).standard_normal((2, 1)),
        learning_rate=0.05,
        n_iterations=2000,
    )

    reg = LinearRegression()
    reg.fit(charge_cpu_std.reshape(-1, 1), consommation)

    assert theta_final[0, 0] == pytest.approx(reg.coef_[0], abs=1e-2)
    assert theta_final[1, 0] == pytest.approx(reg.intercept_, abs=1e-2)

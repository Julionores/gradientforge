import numpy as np
import pytest

from gradientforge import (
    make_design_matrix,
    standardize,
    gradient_descent,
    sigmoid,
    log_loss,
    log_loss_grad,
    predict_class,
    accuracy,
)


def make_network_risk_dataset(seed=0, n_per_class=100):
    rng = np.random.default_rng(seed)
    debit_normal = rng.normal(loc=50, scale=8, size=n_per_class)
    latence_normal = rng.normal(loc=20, scale=5, size=n_per_class)
    debit_risque = rng.normal(loc=15, scale=8, size=n_per_class)
    latence_risque = rng.normal(loc=80, scale=10, size=n_per_class)

    debit = np.concatenate([debit_normal, debit_risque])
    latence = np.concatenate([latence_normal, latence_risque])
    label = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)]).reshape(-1, 1)
    return debit, latence, label


@pytest.fixture
def train_test_split_risk():
    debit, latence, label = make_network_risk_dataset()
    X = make_design_matrix(standardize(debit), standardize(latence))

    rng = np.random.default_rng(0)
    indices = rng.permutation(len(label))
    split = int(0.8 * len(label))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], label[train_idx], label[test_idx]


def test_sigmoid_is_bounded_between_0_and_1():
    z = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    A = sigmoid(z)
    assert np.all((A >= 0) & (A <= 1))
    assert np.isclose(sigmoid(0.0), 0.5)


def test_log_loss_decreases_with_training(train_test_split_risk):
    X_train, _, y_train, _ = train_test_split_risk
    theta_init = np.random.default_rng(0).standard_normal((3, 1))

    theta_final, loss_history = gradient_descent(
        X_train,
        y_train,
        theta_init,
        learning_rate=0.1,
        n_iterations=3000,
        cost_fn=log_loss,
        grad_fn=log_loss_grad,
    )

    assert loss_history[-1] < loss_history[0]
    assert loss_history[-1] < 0.05


def test_classifier_beats_majority_baseline(train_test_split_risk):
    X_train, X_test, y_train, y_test = train_test_split_risk
    theta_init = np.random.default_rng(0).standard_normal((3, 1))

    theta_final, _ = gradient_descent(
        X_train,
        y_train,
        theta_init,
        learning_rate=0.1,
        n_iterations=3000,
        cost_fn=log_loss,
        grad_fn=log_loss_grad,
    )
    predictions = predict_class(X_test, theta_final)
    model_accuracy = accuracy(y_test, predictions)

    classe_majoritaire = round(y_train.mean())
    baseline_predictions = np.full_like(y_test, classe_majoritaire)
    baseline_accuracy = accuracy(y_test, baseline_predictions)

    assert model_accuracy > baseline_accuracy
    assert model_accuracy > 0.9


def test_matches_sklearn_logistic_regression_direction(train_test_split_risk):
    pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression

    X_train, X_test, y_train, y_test = train_test_split_risk
    theta_init = np.random.default_rng(0).standard_normal((3, 1))
    theta_final, _ = gradient_descent(
        X_train,
        y_train,
        theta_init,
        learning_rate=0.1,
        n_iterations=3000,
        cost_fn=log_loss,
        grad_fn=log_loss_grad,
    )

    clf = LogisticRegression()
    clf.fit(X_train[:, :2], y_train.ravel())

    # meme signe (meme logique apprise), scikit-learn regularise donc la magnitude differe
    assert np.sign(theta_final[0, 0]) == np.sign(clf.coef_[0, 0])
    assert np.sign(theta_final[1, 0]) == np.sign(clf.coef_[0, 1])
    assert clf.score(X_test[:, :2], y_test.ravel()) > 0.9

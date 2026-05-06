# Copyright (c) 2026 Malte Londschien
# SPDX-License-Identifier: BSD-3-Clause
#
# Tests adapted from LightGBM's tests/python_package_test/test_engine.py for the
# subset of features rgbm implements. 
#
# rgbm has no `min_data_in_leaf`. Where lgbm relies on its default of 20
# (e.g. test_binary on a small dataset), we substitute
# `min_sum_hessian_in_leaf` — exact for gaussian (hess=1) and a slight
# under-approximation for binary (hess ≤ 0.25).

import numpy as np
import pyarrow as pa
import pytest
import rgbm
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import train_test_split


def _to_batch(X):
    return pa.record_batch(
        {f"f{i}": pa.array(X[:, i], type=pa.float64()) for i in range(X.shape[1])}
    )


def _f64(arr):
    return pa.array(np.asarray(arr, dtype=np.float64), type=pa.float64())


def test_binary():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    m = rgbm.Booster(
        objective="logistic", num_iterations=50, min_sum_hessian_in_leaf=5.0,
    )
    m.fit(rgbm.Dataset(_to_batch(X_train), _f64(y_train)))
    preds = np.asarray(m.predict(_to_batch(X_test)))
    assert log_loss(y_test, preds) < 0.14


def test_regression():
    # Same data as lgbm's tests/utils.py::make_synthetic_regression.
    X, y = make_regression(n_samples=100, n_features=4, n_informative=2, random_state=42)
    y = np.abs(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    m = rgbm.Booster(objective="gaussian", num_iterations=50)
    m.fit(rgbm.Dataset(_to_batch(X_train), _f64(y_train)))
    preds = np.asarray(m.predict(_to_batch(X_test)))
    assert mean_squared_error(y_test, preds) < 343


def test_missing_value_handle():
    rng = np.random.default_rng(0)
    X = np.zeros((100, 1))
    y = np.zeros(100)
    trues = rng.choice(100, size=20, replace=False)
    X[trues, 0] = np.nan
    y[trues] = 1
    batch = _to_batch(X)
    m = rgbm.Booster(objective="gaussian", num_iterations=20)
    m.fit(rgbm.Dataset(batch, _f64(y)))
    preds = np.asarray(m.predict(batch))
    assert mean_squared_error(y, preds) < 0.005


def test_missing_value_handle_more_na():
    rng = np.random.default_rng(0)
    X = np.ones((100, 1))
    y = np.ones(100)
    trues = rng.choice(100, size=80, replace=False)
    X[trues, 0] = np.nan
    y[trues] = 0
    batch = _to_batch(X)
    m = rgbm.Booster(objective="gaussian", num_iterations=20)
    m.fit(rgbm.Dataset(batch, _f64(y)))
    preds = np.asarray(m.predict(batch))
    assert mean_squared_error(y, preds) < 0.005


def test_missing_value_handle_na():
    # Tiny dataset, single tree, lr=1: model recovers labels exactly,
    # NaN routes to the side with matching gradient.
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, np.nan]).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1], dtype=np.float64)
    batch = _to_batch(x)
    m = rgbm.Booster(
        objective="gaussian", num_iterations=1, learning_rate=1.0, max_leaves=2,
    )
    m.fit(rgbm.Dataset(batch, _f64(y), min_data_in_bin=1))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, y, atol=1e-6)


def test_categorical_handle():
    # Alternating-cluster categorical: single tree, lr=1, two leaves
    # learns the partition perfectly.
    cats = pa.array([str(i) for i in range(8)]).dictionary_encode()
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float64)
    batch = pa.record_batch({"cat": cats})
    m = rgbm.Booster(
        objective="gaussian", num_iterations=1, learning_rate=1.0, max_leaves=2,
    )
    m.fit(rgbm.Dataset(batch, _f64(y), min_data_in_bin=1))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, y)


def test_categorical_handle_na():
    cats = pa.array(["a", None, "a", None, "a", None]).dictionary_encode()
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.float64)
    batch = pa.record_batch({"cat": cats})
    m = rgbm.Booster(
        objective="gaussian", num_iterations=1, learning_rate=1.0, max_leaves=2,
    )
    m.fit(rgbm.Dataset(batch, _f64(y), min_data_in_bin=1))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, y)


@pytest.mark.parametrize("y_true,expected", [
    ([0.0, 10.0, 0.0, 10.0], 5.0),
    ([0.0, 1.0, 2.0, 3.0], 1.5),
    ([-1.0, 1.0, -2.0, 2.0], 0.0),
])
def test_constant_features_regression(y_true, expected):
    X = np.ones((len(y_true), 1))
    batch = _to_batch(X)
    m = rgbm.Booster(objective="gaussian", num_iterations=2, learning_rate=1.0, max_leaves=2)
    m.fit(rgbm.Dataset(batch, _f64(y_true), min_data_in_bin=1))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, expected)


def test_small_max_bin():
    rng = np.random.default_rng(0)
    y = rng.choice([0, 1], 100).astype(np.float64)
    X = np.ones((100, 1))
    X[:30, 0] = -1
    X[60:, 0] = 2
    m = rgbm.Booster(objective="logistic", num_iterations=5)
    m.fit(rgbm.Dataset(_to_batch(X), _f64(y), max_bin=2))
    X[0, 0] = np.nan
    m = rgbm.Booster(objective="logistic", num_iterations=5)
    m.fit(rgbm.Dataset(_to_batch(X), _f64(y), max_bin=3))

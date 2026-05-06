# Copyright (c) 2026 Malte Londschien
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pyarrow as pa
import pytest
import rgbm
from sklearn.datasets import make_regression


def _to_batch(X):
    return pa.record_batch(
        {f"f{i}": pa.array(X[:, i], type=pa.float64()) for i in range(X.shape[1])}
    )


def _f64(arr):
    return pa.array(np.asarray(arr, dtype=np.float64), type=pa.float64())


@pytest.mark.parametrize("objective", ["logistic", "probit"])
def test_predictions_are_probabilities(objective):
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    batch = pa.record_batch({"x": _f64(x)})
    m = rgbm.Booster(objective=objective, num_iterations=10)
    m.fit(rgbm.Dataset(batch, _f64((x > 0).astype(np.float64))))
    preds = np.asarray(m.predict(batch))
    assert (preds >= 0).all() and (preds <= 1).all()


def test_determinism():
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    batch = _to_batch(X)
    y_arr = _f64(y)

    m1 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    m1.fit(rgbm.Dataset(batch, y_arr))
    m2 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    m2.fit(rgbm.Dataset(batch, y_arr))
    np.testing.assert_array_equal(np.asarray(m1.predict(batch)), np.asarray(m2.predict(batch)))


def test_lambda_l2_shrinks_predictions():
    X, y = make_regression(n_samples=300, n_features=3, random_state=0)
    batch = _to_batch(X)
    y_arr = _f64(y)

    m_no = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3, lambda_l2=0.0)
    m_no.fit(rgbm.Dataset(batch, y_arr))
    m_l2 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3, lambda_l2=1e4)
    m_l2.fit(rgbm.Dataset(batch, y_arr))
    assert np.asarray(m_l2.predict(batch)).std() < np.asarray(m_no.predict(batch)).std()


def test_max_bin_affects_granularity():
    X, y = make_regression(n_samples=500, n_features=1, random_state=0)
    batch = _to_batch(X)
    y_arr = _f64(y)

    m_few = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_few.fit(rgbm.Dataset(batch, y_arr, max_bin=4))
    m_many = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_many.fit(rgbm.Dataset(batch, y_arr, max_bin=255))
    assert len(np.unique(np.asarray(m_few.predict(batch)))) < len(np.unique(np.asarray(m_many.predict(batch))))


def test_sample_weights():
    # Weights enter via the base score (weighted mean) and gradient aggregation,
    # so a heavily skewed weight vector must change predictions.
    X, y = make_regression(n_samples=200, n_features=3, random_state=0)
    batch = _to_batch(X)
    y_arr = _f64(y)
    w = np.where(np.arange(200) < 100, 10.0, 1.0)

    m_no = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_no.fit(rgbm.Dataset(batch, y_arr))
    m_w = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_w.fit(rgbm.Dataset(batch, y_arr, weights=_f64(w)))
    assert not np.allclose(np.asarray(m_no.predict(batch)), np.asarray(m_w.predict(batch)))


def test_min_gain_to_split_blocks_all_splits():
    # A very large min_gain_to_split rejects every candidate split, so every
    # tree is a single leaf with value 0 and predictions equal the base score.
    X, y = make_regression(n_samples=300, n_features=3, random_state=0)
    batch = _to_batch(X)
    m = rgbm.Booster(objective="gaussian", num_iterations=10, min_gain_to_split=1e18)
    m.fit(rgbm.Dataset(batch, _f64(y)))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, y.mean())


def test_max_depth_limits_leaf_count():
    X, y = make_regression(n_samples=500, n_features=5, random_state=0)
    batch = _to_batch(X)
    y_arr = _f64(y)

    m_shallow = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=2)
    m_shallow.fit(rgbm.Dataset(batch, y_arr))
    m_deep = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=8)
    m_deep.fit(rgbm.Dataset(batch, y_arr))
    assert len(np.unique(np.asarray(m_shallow.predict(batch)))) < len(np.unique(np.asarray(m_deep.predict(batch))))

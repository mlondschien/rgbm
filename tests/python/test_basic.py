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


@pytest.mark.parametrize("objective", ["logistic", "probit"])
def test_predictions_are_probabilities(objective):
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    batch = pa.record_batch({"x": pa.array(x, type=pa.float64())})
    m = rgbm.Booster(objective=objective, num_iterations=10)
    m.fit(rgbm.Dataset(batch, (x > 0).astype(np.float64)))
    preds = np.asarray(m.predict(batch))
    assert (preds >= 0).all() and (preds <= 1).all()


def test_determinism():
    X, y = make_regression(n_samples=200, n_features=4, random_state=0)
    batch = _to_batch(X)

    m1 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    m1.fit(rgbm.Dataset(batch, y))
    m2 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    m2.fit(rgbm.Dataset(batch, y))
    np.testing.assert_array_equal(
        np.asarray(m1.predict(batch)), np.asarray(m2.predict(batch))
    )


def test_lambda_l2_shrinks_predictions():
    X, y = make_regression(n_samples=300, n_features=3, random_state=0)
    batch = _to_batch(X)

    m_no = rgbm.Booster(
        objective="gaussian", num_iterations=20, max_depth=3, lambda_l2=0.0
    )
    m_no.fit(rgbm.Dataset(batch, y))
    m_l2 = rgbm.Booster(
        objective="gaussian", num_iterations=20, max_depth=3, lambda_l2=1e4
    )
    m_l2.fit(rgbm.Dataset(batch, y))
    assert np.asarray(m_l2.predict(batch)).std() < np.asarray(m_no.predict(batch)).std()


def test_max_bin_affects_granularity():
    X, y = make_regression(n_samples=500, n_features=1, random_state=0)
    batch = _to_batch(X)

    m_few = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_few.fit(rgbm.Dataset(batch, y, max_bin=4))
    m_many = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_many.fit(rgbm.Dataset(batch, y, max_bin=255))
    assert len(np.unique(np.asarray(m_few.predict(batch)))) < len(
        np.unique(np.asarray(m_many.predict(batch)))
    )


def test_sample_weights():
    # Weights enter via the base score (weighted mean) and gradient aggregation,
    # so a heavily skewed weight vector must change predictions.
    X, y = make_regression(n_samples=200, n_features=3, random_state=0)
    batch = _to_batch(X)
    w = np.where(np.arange(200) < 100, 10.0, 1.0)

    m_no = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_no.fit(rgbm.Dataset(batch, y))
    m_w = rgbm.Booster(objective="gaussian", num_iterations=10, max_depth=3)
    m_w.fit(rgbm.Dataset(batch, y, weights=w))
    assert not np.allclose(
        np.asarray(m_no.predict(batch)), np.asarray(m_w.predict(batch))
    )


def test_min_gain_to_split_blocks_all_splits():
    # A very large min_gain_to_split rejects every candidate split, so every
    # tree is a single leaf with value 0 and predictions equal the base score.
    X, y = make_regression(n_samples=300, n_features=3, random_state=0)
    batch = _to_batch(X)
    m = rgbm.Booster(objective="gaussian", num_iterations=10, min_gain_to_split=1e18)
    m.fit(rgbm.Dataset(batch, y))
    preds = np.asarray(m.predict(batch))
    np.testing.assert_allclose(preds, y.mean())


def test_max_depth_limits_leaf_count():
    X, y = make_regression(n_samples=500, n_features=5, random_state=0)
    batch = _to_batch(X)

    m_shallow = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=2)
    m_shallow.fit(rgbm.Dataset(batch, y))
    m_deep = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=8)
    m_deep.fit(rgbm.Dataset(batch, y))
    assert len(np.unique(np.asarray(m_shallow.predict(batch)))) < len(
        np.unique(np.asarray(m_deep.predict(batch)))
    )


def _mixed_batch(numerical_dtype=pa.float64(), dict_value_type=pa.string()):
    # Numerical values are rounded to f32 so the f32 re-encoding is bit-identical.
    rng = np.random.default_rng(0)
    num_rows = 300
    features = rng.normal(size=(num_rows, 4)).astype(np.float32).astype(np.float64)
    categories = ["a", "b", "c", "d"]
    category_indices = rng.integers(0, len(categories), size=num_rows).astype(np.int32)
    labels = (
        features[:, 0]
        + (category_indices == 0).astype(np.float64)
        + 0.1 * rng.normal(size=num_rows)
    )
    columns = {
        f"f{i}": pa.array(features[:, i], type=numerical_dtype)
        for i in range(features.shape[1])
    }
    columns["c"] = pa.DictionaryArray.from_arrays(
        pa.array(category_indices, type=pa.uint32()),
        pa.array(categories, type=dict_value_type),
    )
    return pa.record_batch(columns), labels


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"numerical_dtype": pa.float32()}, id="float32"),
        pytest.param({"dict_value_type": pa.large_string()}, id="large_utf8"),
        pytest.param({"dict_value_type": pa.string_view()}, id="utf8_view"),
    ],
)
def test_alternate_arrow_dtypes_match_float64_utf8(kwargs):
    batch_f64_utf8, labels = _mixed_batch()
    batch_alternate, _ = _mixed_batch(**kwargs)

    model_f64_utf8 = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    model_f64_utf8.fit(rgbm.Dataset(batch_f64_utf8, labels))
    model_alternate = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    model_alternate.fit(rgbm.Dataset(batch_alternate, labels))
    np.testing.assert_array_equal(
        np.asarray(model_f64_utf8.predict(batch_f64_utf8)),
        np.asarray(model_alternate.predict(batch_alternate)),
    )


def test_polars_categorical_dataframe():
    # End-to-end: a polars DataFrame with a Categorical column (Dict<_, LargeUtf8>
    # in arrow) and an f32 column trains and predicts without manual conversion.
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(0)
    num_rows = 300
    groups = rng.choice(["a", "b", "c"], size=num_rows)
    df = pl.DataFrame(
        {
            "x": rng.normal(size=num_rows).astype(np.float32),
            "g": pl.Series(groups, dtype=pl.Categorical),
        }
    )
    labels = (groups == "a").astype(np.float64) + rng.normal(scale=0.1, size=num_rows)

    model = rgbm.Booster(objective="gaussian", num_iterations=20, max_depth=3)
    model.fit(rgbm.Dataset(df, labels))
    predictions = np.asarray(model.predict(df))
    assert predictions[groups == "a"].mean() > predictions[groups == "c"].mean()

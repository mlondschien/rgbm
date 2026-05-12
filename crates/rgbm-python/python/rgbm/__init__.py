# Copyright (c) 2026 Malte Londschien
# SPDX-License-Identifier: BSD-3-Clause

"""rgbm: a fast and lean gradient boosting machine."""

import numpy as np
import pyarrow as pa

from rgbm._core import Dataset as _Dataset, Booster as _Booster

__all__ = ["Dataset", "Booster"]


def _to_record_batch(x):
    """Coerce ``x`` to ``pyarrow.RecordBatch``.

    Accepts a polars ``DataFrame`` (or any object exposing the Arrow
    PyCapsule ``__arrow_c_stream__`` interface), a ``pyarrow.RecordBatch``,
    or a ``pyarrow.Table``.
    """
    if isinstance(x, pa.RecordBatch):
        return x
    if isinstance(x, pa.Table):
        return pa.concat_batches(x.to_batches())
    if hasattr(x, "__arrow_c_stream__"):
        return pa.concat_batches(pa.table(x).to_batches())
    raise TypeError(
        f"unsupported type for feature matrix: {type(x).__name__}; expected "
        "pyarrow.RecordBatch, pyarrow.Table, or polars.DataFrame"
    )


def _to_array(y):
    """Coerce ``y`` to ``pyarrow.Array`` (Float64).

    Accepts a polars ``Series``, a numpy array, a ``pyarrow.Array``, or
    anything ``np.asarray`` can convert (lists, etc.).
    """
    if isinstance(y, pa.Array):
        return y
    return pa.array(np.asarray(y, dtype=np.float64), type=pa.float64())


class Dataset:
    """A binned representation of a feature matrix and labels for training.

    Numerical columns are bucketed into ``max_bin`` bins via greedy quantile
    binning. Categorical (Arrow ``Dictionary``) columns are mapped to bin
    indices per category.

    Parameters
    ----------
    x : polars.DataFrame, pyarrow.RecordBatch, or pyarrow.Table
        Feature matrix. Each column must be numerical (Float64 / Float32) or
        Categorical (Arrow ``Dictionary`` with string values).
    y : polars.Series, pyarrow.Array, or numpy.ndarray
        Float64 labels.
    weights : polars.Series, pyarrow.Array, or numpy.ndarray, optional
        Per-row weights (Float64). Defaults to uniform weights.
    offsets : polars.Series, pyarrow.Array, or numpy.ndarray, optional
        Per-row baseline (Float64) added to the raw score during fit and
        predict.
    max_bin : int, default 255
        Maximum number of bins per feature, including the missing/sentinel
        bin. Must satisfy ``max_bin <= 255``.
    min_data_in_bin : int, default 3
        Minimum number of rows that must accumulate before opening a new bin
        during quantile binning of numerical features.
    n_jobs : int, default -1
        Number of threads used for binning. ``-1`` uses all logical cores.
    seed : int, default 0
        Seed for the row subsample used to determine bin boundaries on large
        datasets (>200,000 rows).
    """

    def __init__(self, x, y, weights=None, offsets=None, max_bin=255, min_data_in_bin=3, n_jobs=-1, seed=0):
        self._inner = _Dataset(
            _to_record_batch(x),
            _to_array(y),
            _to_array(weights) if weights is not None else None,
            _to_array(offsets) if offsets is not None else None,
            max_bin,
            min_data_in_bin,
            n_jobs,
            seed,
        )


class Booster:
    """Gradient-boosted decision tree model.

    Parameters
    ----------
    objective : {"gaussian", "logistic", "probit", "poisson"}, default "gaussian"
        Loss function. ``gaussian`` for regression, ``logistic`` and
        ``probit`` for binary classification with labels in ``{0, 1}``,
        ``poisson`` for non-negative count regression.
    num_iterations : int, default 100
        Number of boosting rounds (trees).
    learning_rate : float, default 0.1
        Multiplier applied to each tree's leaf values before adding to the
        ensemble.
    max_depth : int, default 6
        Maximum tree depth.
    max_leaves : int, default 31
        Maximum number of leaves per tree.
    min_sum_hessian_in_leaf : float, default 1e-3
        Minimum sum of hessians required for a leaf to be split.
    min_gain_to_split : float, default 0.0
        Minimum split gain required for a leaf to be split.
    lambda_l1 : float, default 0.0
        L1 regularization on leaf values.
    lambda_l2 : float, default 0.0
        L2 regularization on leaf values.
    leaf_wise : bool, default True
        If True, grow trees by splitting the highest-gain leaf first
        (LightGBM-style). If False, grow level-wise: split shallowest leaves
        first, ties broken by gain (xgboost ``grow_policy=depthwise``).
    n_jobs : int, default -1
        Number of threads used for fitting and prediction. ``-1`` uses all
        logical cores.
    """

    def __init__(
        self,
        objective="gaussian",
        num_iterations=100,
        learning_rate=0.1,
        max_depth=6,
        min_sum_hessian_in_leaf=1e-3,
        min_gain_to_split=0.0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        max_leaves=31,
        leaf_wise=True,
        n_jobs=-1,
    ):
        self._inner = _Booster(
            objective,
            num_iterations,
            learning_rate,
            max_depth,
            min_sum_hessian_in_leaf,
            min_gain_to_split,
            lambda_l1,
            lambda_l2,
            max_leaves,
            leaf_wise,
            n_jobs,
        )

    def fit(self, dataset):
        """Fit the booster on a Dataset.

        Parameters
        ----------
        dataset : Dataset
            Training dataset built via :class:`Dataset`.
        """
        self._inner.fit(dataset._inner)

    def predict(self, x, offsets=None):
        """Predict on a feature matrix.

        Parameters
        ----------
        x : polars.DataFrame, pyarrow.RecordBatch, or pyarrow.Table
            Feature matrix with the same schema as the training data.
        offsets : polars.Series, pyarrow.Array, or numpy.ndarray, optional
            Per-row baseline (Float64) added to the raw score before applying
            the objective's link function.

        Returns
        -------
        numpy.ndarray of float64
            Per-row predictions. For ``gaussian``, raw scores. For
            ``logistic`` and ``probit``, probabilities in ``[0, 1]``. For
            ``poisson``, expected counts.
        """
        offsets_arr = _to_array(offsets) if offsets is not None else None
        return self._inner.predict(_to_record_batch(x), offsets_arr)

    def model_to_string(self):
        """Serialize the model to a LightGBM-compatible ``model.txt`` (v4) string.

        The returned string can be loaded back via
        ``lightgbm.Booster(model_str=...)`` for prediction. Useful for
        interoperability with downstream tooling that expects lgbm models.
        """
        return self._inner.model_to_string()

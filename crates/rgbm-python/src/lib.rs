// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use arrow::array::{ArrayData, Float64Array, RecordBatch};
use arrow::pyarrow::FromPyArrow;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::rgbm::booster::Booster;
use ::rgbm::dataset::Dataset;
use ::rgbm::objective::{Gaussian, Logistic, Objective, Probit};
use ::rgbm::parameters::{BoosterParameters, DatasetParameters};

/// A binned representation of a feature matrix and labels for training.
///
/// Numerical columns are bucketed into ``max_bin`` bins via greedy quantile
/// binning. Categorical columns (Arrow ``Dictionary``) are mapped to bin
/// indices per category. Float32 numerical columns and dictionaries with
/// ``LargeUtf8`` / ``Utf8View`` values are accepted and cast internally, so
/// polars and pandas DataFrames can be passed directly via ``df.to_arrow()``.
///
/// Parameters
/// ----------
/// x : pyarrow.RecordBatch
///     Feature matrix. Each column must be Float64, Float32, or a Dictionary
///     with string-typed values (Utf8, LargeUtf8, or Utf8View). Integer
///     columns are not accepted due to ambiguity around whether they should
///     be treated as numerical or categorical.
/// y : pyarrow.Array
///     Float64 labels.
/// weights : pyarrow.Array, optional
///     Per-row weights (Float64). Defaults to uniform weights if ``None``.
/// max_bin : int, default 255
///     Maximum number of bins per feature, including the missing/sentinel bin.
///     Must satisfy ``max_bin <= 255``.
/// min_data_in_bin : int, default 3
///     Minimum number of rows that must accumulate before opening a new bin
///     during quantile binning of numerical features.
/// n_jobs : int, default -1
///     Number of threads used for binning. ``-1`` uses all logical cores.
/// seed : int, default 0
///     Seed for the row subsample used to determine bin boundaries on large
///     datasets (>200,000 rows).
#[pyclass(name = "Dataset")]
struct PyDataset {
    inner: Dataset,
}

#[pymethods]
impl PyDataset {
    #[new]
    #[pyo3(signature = (x, y, weights=None, max_bin=255, min_data_in_bin=3, n_jobs=-1, seed=0))]
    fn new(
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        weights: Option<&Bound<'_, PyAny>>,
        max_bin: usize,
        min_data_in_bin: usize,
        n_jobs: isize,
        seed: u64,
    ) -> PyResult<Self> {
        let batch = RecordBatch::from_pyarrow_bound(x)?;
        let labels = Float64Array::from(ArrayData::from_pyarrow_bound(y)?);
        let weights = if let Some(w) = weights {
            Some(Float64Array::from(ArrayData::from_pyarrow_bound(w)?))
        } else {
            None
        };
        let params = DatasetParameters { max_bin, min_data_in_bin, n_jobs, seed };
        let inner = Dataset::from_arrow(&batch, &labels, weights.as_ref(), &params);
        Ok(Self { inner })
    }
}

/// Gradient-boosted decision tree model.
///
/// Parameters
/// ----------
/// objective : {"gaussian", "logistic", "probit"}, default "gaussian"
///     Loss function. ``gaussian`` for regression, ``logistic`` and ``probit``
///     for binary classification with labels in ``{0, 1}``.
/// num_iterations : int, default 100
///     Number of boosting rounds (trees).
/// learning_rate : float, default 0.1
///     Multiplier applied to each tree's leaf values before adding to the
///     ensemble.
/// max_depth : int, default 6
///     Maximum tree depth.
/// max_leaves : int, default 31
///     Maximum number of leaves per tree.
/// min_sum_hessian_in_leaf : float, default 1e-3
///     Minimum sum of hessians required for a leaf to be split.
/// min_gain_to_split : float, default 0.0
///     Minimum split gain required for a leaf to be split.
/// lambda_l1 : float, default 0.0
///     L1 regularization on leaf values.
/// lambda_l2 : float, default 0.0
///     L2 regularization on leaf values.
/// leaf_wise : bool, default True
///     If True, grow trees by splitting the highest-gain leaf first
///     (LightGBM-style). If False, grow level-wise: split shallowest leaves
///     first, ties broken by gain (xgboost ``grow_policy=depthwise``).
/// n_jobs : int, default -1
///     Number of threads used for fitting and prediction. ``-1`` uses all
///     logical cores.
#[pyclass(name = "Booster")]
struct PyBooster {
    booster: Booster,
}

#[pymethods]
impl PyBooster {
    #[new]
    #[pyo3(signature = (
        objective = "gaussian",
        num_iterations = 100,
        learning_rate = 0.1,
        max_depth = 6,
        min_sum_hessian_in_leaf = 1e-3,
        min_gain_to_split = 0.0,
        lambda_l1 = 0.0,
        lambda_l2 = 0.0,
        max_leaves = 31,
        leaf_wise = true,
        n_jobs = -1,
    ))]
    fn new(
        objective: &str,
        num_iterations: usize,
        learning_rate: f64,
        max_depth: usize,
        min_sum_hessian_in_leaf: f64,
        min_gain_to_split: f64,
        lambda_l1: f64,
        lambda_l2: f64,
        max_leaves: usize,
        leaf_wise: bool,
        n_jobs: isize,
    ) -> PyResult<Self> {
        let obj: Box<dyn Objective> = match objective {
            "gaussian" => Box::new(Gaussian),
            "logistic" => Box::new(Logistic),
            "probit"   => Box::new(Probit),
            _ => return Err(PyValueError::new_err(format!(
                "unknown objective '{objective}'; expected one of: gaussian, logistic, probit"
            ))),
        };

        Ok(Self {
            booster: Booster::new(BoosterParameters {
                num_iterations, learning_rate, max_depth,
                min_sum_hessian_in_leaf, min_gain_to_split,
                lambda_l1, lambda_l2, max_leaves, leaf_wise, n_jobs,
            }, obj),
        })
    }

    /// Fit the booster on a Dataset.
    ///
    /// Parameters
    /// ----------
    /// dataset : Dataset
    ///     Training dataset built via :class:`Dataset`.
    fn fit(&mut self, py: Python<'_>, dataset: &PyDataset) -> PyResult<()> {
        py.allow_threads(|| self.booster.fit(&dataset.inner));
        Ok(())
    }

    /// Serialize the model to a LightGBM-compatible ``model.txt`` (v4) string.
    ///
    /// The returned string can be loaded back via
    /// ``lightgbm.Booster(model_str=...)`` for prediction. Useful for
    /// interoperability with downstream tooling that expects lgbm models.
    fn model_to_string(&self) -> String {
        self.booster.model_to_string()
    }

    /// Predict on a feature matrix.
    ///
    /// Parameters
    /// ----------
    /// x : pyarrow.RecordBatch
    ///     Feature matrix with the same schema as the training data. Float32
    ///     and non-Utf8 dictionary value types are cast internally.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray of float64
    ///     Per-row predictions. For ``gaussian``, raw scores. For ``logistic``
    ///     and ``probit``, probabilities in ``[0, 1]``.
    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let batch = RecordBatch::from_pyarrow_bound(x)?;
        let result = py.allow_threads(|| self.booster.predict(&batch));
        Ok(PyArray1::from_slice(py, result.values()).into())
    }
}

#[pymodule]
fn rgbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDataset>()?;
    m.add_class::<PyBooster>()?;
    Ok(())
}

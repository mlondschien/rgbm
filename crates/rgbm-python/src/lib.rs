use arrow::array::{ArrayData, Float64Array, RecordBatch};
use arrow::pyarrow::FromPyArrow;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ::rgbm::booster::Booster;
use ::rgbm::dataset::Dataset;
use ::rgbm::objective::{BinaryLogloss, Objective, Probit, SquaredLoss};
use ::rgbm::parameters::Parameters;

#[pyclass]
struct GradientBooster {
    booster: Booster,
}

#[pymethods]
impl GradientBooster {
    #[new]
    #[pyo3(signature = (
        objective = "squared_loss",
        num_iterations = 100,
        learning_rate = 0.1,
        max_depth = 6,
        min_sum_hessian_in_leaf = 1e-3,
        lambda_l1 = 0.0,
        lambda_l2 = 0.0,
        max_bin = 255,
        min_data_in_bin = 3,
        max_leaves = 31,
        leaf_wise = true,
    ))]
    fn new(
        objective: &str,
        num_iterations: usize,
        learning_rate: f64,
        max_depth: usize,
        min_sum_hessian_in_leaf: f64,
        lambda_l1: f64,
        lambda_l2: f64,
        max_bin: usize,
        min_data_in_bin: usize,
        max_leaves: usize,
        leaf_wise: bool,
    ) -> PyResult<Self> {
        let obj: Box<dyn Objective> = match objective {
            "squared_loss"   => Box::new(SquaredLoss),
            "binary_logloss" => Box::new(BinaryLogloss),
            "probit"         => Box::new(Probit),
            _ => return Err(PyValueError::new_err(format!(
                "unknown objective '{objective}'; expected one of: squared_loss, binary_logloss, probit"
            ))),
        };

        Ok(Self {
            booster: Booster::new(Parameters {
                num_iterations, learning_rate, max_depth,
                min_sum_hessian_in_leaf,
                lambda_l1, lambda_l2, max_bin, min_data_in_bin, max_leaves, leaf_wise,
            }, obj),
        })
    }

    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<()> {
        let batch = RecordBatch::from_pyarrow_bound(x)?;
        let labels = Float64Array::from(ArrayData::from_pyarrow_bound(y)?);
        let p = &self.booster.parameters;
        let dataset = Dataset::from_arrow(&batch, &labels, None, p.max_bin, p.min_data_in_bin);
        py.allow_threads(|| self.booster.fit(&dataset));
        Ok(())
    }

    fn predict(&self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let batch = RecordBatch::from_pyarrow_bound(x)?;
        let result = py.allow_threads(|| self.booster.predict(&batch));
        Ok(PyArray1::from_slice(py, result.values()).into())
    }
}

#[pymodule]
fn rgbm(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<GradientBooster>()?;
    Ok(())
}

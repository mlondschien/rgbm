// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use arrow::array::{Float64Array, PrimitiveArray};
use arrow::datatypes::Float64Type;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::bin::FeatureBinner;
use crate::dataset::Dataset;
use crate::parameters::BoosterParameters;
use crate::objective::Objective;
use crate::tree::{Tree, TreeWorkspace};
use crate::utils::build_thread_pool;

pub struct Booster {
    pub parameters: BoosterParameters,
    pub objective: Box<dyn Objective>,
    pub trees: Vec<Tree>,
    pub base_score: f64,
    pub feature_binners: Vec<FeatureBinner>,
}

impl Booster {
    pub fn new(parameters: BoosterParameters, objective: Box<dyn Objective>) -> Self {
        Self {
            parameters,
            objective,
            trees: Vec::new(),
            base_score: 0.0,
            feature_binners: Vec::new(),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        self.feature_binners = dataset.feature_binners.clone();
        let labels = dataset.labels.values();
        self.base_score = self.objective.initial_score(labels);

        let mut scores = vec![self.base_score; dataset.num_rows];
        let mut grad_hess = vec![[0.0f32; 2]; dataset.num_rows];
        let mut workspace = TreeWorkspace::new(dataset.num_rows);

        let pool = build_thread_pool(self.parameters.n_jobs);

        self.trees.clear();

        for _ in 0..self.parameters.num_iterations {
            self.objective.gradient_hessian(labels, &scores, &mut grad_hess, pool.as_ref());

            let mut tree = Tree::new(self.parameters.max_leaves);
            tree.fit(dataset, &grad_hess, &self.parameters, pool.as_ref(), &mut workspace);

            match &pool {
                Some(pool) => {
                    let nodes = &tree.nodes;
                    pool.install(|| {
                        scores.par_iter_mut().zip(workspace.leaf_indices.par_iter()).for_each(|(score, &leaf_idx)| {
                            *score += nodes[leaf_idx as usize].value();
                        });
                    });
                }
                None => {
                    for (score, &leaf_idx) in scores.iter_mut().zip(&workspace.leaf_indices) {
                        *score += tree.nodes[leaf_idx as usize].value();
                    }
                }
            }

            self.trees.push(tree);
        }
    }

    pub fn predict(&self, batch: &RecordBatch) -> Float64Array {
        let num_cols = batch.num_columns();
        let mut numeric_columns: Vec<Option<&PrimitiveArray<Float64Type>>> = vec![None; num_cols];
        let mut categorical_columns: Vec<Option<Vec<u8>>> = vec![None; num_cols];
        for (i, col) in batch.columns().iter().enumerate() {
            if let Some(arr) = col.as_any().downcast_ref::<PrimitiveArray<Float64Type>>() {
                numeric_columns[i] = Some(arr);
            } else if matches!(col.data_type(), arrow::datatypes::DataType::Dictionary(_, _)) {
                categorical_columns[i] = Some(self.feature_binners[i].apply(col.as_ref()));
            }
        }

        let num_rows = batch.num_rows();
        let mut scores = vec![self.base_score; num_rows];

        let pool = build_thread_pool(self.parameters.n_jobs);
        match &pool {
            Some(pool) => pool.install(|| {
                scores.par_iter_mut().enumerate().for_each(|(row, score)| {
                    for tree in &self.trees {
                        *score += tree.predict_row(row, &numeric_columns, &categorical_columns);
                    }
                });
            }),
            None => {
                for (row, score) in scores.iter_mut().enumerate() {
                    for tree in &self.trees {
                        *score += tree.predict_row(row, &numeric_columns, &categorical_columns);
                    }
                }
            }
        }

        Float64Array::from_iter(scores.iter().map(|&s| self.objective.prediction(s)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use crate::dataset::Dataset;
    use crate::objective::{BinaryLogloss, Probit, SquaredLoss};
    use crate::parameters::{BoosterParameters, DatasetParameters};

    fn make_dataset(x: Vec<f64>, y: Vec<f64>) -> (Dataset, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(x))]).unwrap();
        let labels = Float64Array::from(y);
        let dataset = Dataset::from_arrow(&batch, &labels, None, &DatasetParameters { min_data_in_bin: 1, ..DatasetParameters::default() });
        (dataset, batch)
    }

    fn test_params() -> BoosterParameters {
        BoosterParameters { num_iterations: 20, min_sum_hessian_in_leaf: 0.0, ..BoosterParameters::default() }
    }

    fn mse(preds: &Float64Array, labels: &[f64]) -> f64 {
        preds.values().iter().zip(labels).map(|(p, y)| (p - y).powi(2)).sum::<f64>() / labels.len() as f64
    }

    #[test]
    fn test_base_score_is_mean() {
        let y = vec![0.0, 1.0, 2.0, 3.0];
        let (dataset, _) = make_dataset(vec![0.0, 1.0, 2.0, 3.0], y.clone());
        let mut booster = Booster::new(BoosterParameters { num_iterations: 0, ..test_params() }, Box::new(SquaredLoss));
        booster.fit(&dataset);
        assert!((booster.base_score - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_fit_regression_reduces_mse() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let (dataset, batch) = make_dataset(x, y.clone());

        let variance = y.iter().map(|&yi| (yi - 1.5f64).powi(2)).sum::<f64>() / n as f64;

        let mut booster = Booster::new(test_params(), Box::new(SquaredLoss));
        booster.fit(&dataset);
        let preds = booster.predict(&batch);
        assert!(mse(&preds, &y) < variance * 0.01);
    }

    #[test]
    fn test_fit_probit() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| if xi > 0.5 { 1.0 } else { 0.0 }).collect();
        let (dataset, batch) = make_dataset(x, y.clone());

        let mut booster = Booster::new(test_params(), Box::new(Probit));
        booster.fit(&dataset);
        let preds = booster.predict(&batch);

        let correct = preds.values().iter().zip(&y)
            .filter(|&(&p, &yi)| (p > 0.5) == (yi > 0.5))
            .count();
        assert!(correct as f64 / n as f64 > 0.95);
    }

    #[test]
    fn test_fit_classification_reduces_logloss() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| if xi > 0.5 { 1.0 } else { 0.0 }).collect();
        let (dataset, batch) = make_dataset(x, y.clone());

        let mut booster = Booster::new(test_params(), Box::new(BinaryLogloss));
        booster.fit(&dataset);
        let preds = booster.predict(&batch);

        // predictions should be on the correct side of 0.5
        let correct = preds.values().iter().zip(&y)
            .filter(|&(&p, &yi)| (p > 0.5) == (yi > 0.5))
            .count();
        assert!(correct as f64 / n as f64 > 0.95);
    }
}

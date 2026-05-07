// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use arrow::array::Float64Array;
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
    pub feature_names: Vec<String>,
}

impl Booster {
    pub fn new(parameters: BoosterParameters, objective: Box<dyn Objective>) -> Self {
        Self {
            parameters,
            objective,
            trees: Vec::new(),
            base_score: 0.0,
            feature_binners: Vec::new(),
            feature_names: Vec::new(),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        self.feature_binners = dataset.feature_binners.clone();
        self.feature_names = dataset.feature_names.clone();
        let labels = dataset.labels.values();
        let weights = dataset.weights.as_ref().map(|w| w.values().as_ref());
        self.base_score = self.objective.initial_score(labels, weights);

        let mut scores = vec![self.base_score; dataset.num_rows];
        let mut grad_hess = vec![[0.0f32; 2]; dataset.num_rows];
        let mut workspace = TreeWorkspace::new(dataset.num_rows);

        let pool = build_thread_pool(self.parameters.n_jobs);

        self.trees.clear();

        for _ in 0..self.parameters.num_iterations {
            self.objective.gradient_hessian(labels, &scores, weights, &mut grad_hess, pool.as_ref());

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
        // Bin every column up-front. FeatureBinner::apply handles Float32 and the
        // various dictionary value types via internal casts.
        let columns: Vec<Vec<u8>> = self.feature_binners.iter().zip(batch.columns())
            .map(|(b, col)| b.apply(col.as_ref()))
            .collect();
        let column_refs: Vec<&[u8]> = columns.iter().map(|c| c.as_slice()).collect();
        let sentinels: Vec<u8> = self.feature_binners.iter().map(|b| (b.num_bins() - 1) as u8).collect();

        let num_rows = batch.num_rows();
        let mut scores = vec![self.base_score; num_rows];

        let pool = build_thread_pool(self.parameters.n_jobs);
        match &pool {
            Some(pool) => pool.install(|| {
                scores.par_iter_mut().enumerate().for_each(|(row, score)| {
                    for tree in &self.trees {
                        *score += tree.predict_row(row, &column_refs, &sentinels);
                    }
                });
            }),
            None => {
                for (row, score) in scores.iter_mut().enumerate() {
                    for tree in &self.trees {
                        *score += tree.predict_row(row, &column_refs, &sentinels);
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
    use crate::objective::{Gaussian, Logistic, Probit};
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
        let mut booster = Booster::new(BoosterParameters { num_iterations: 0, ..test_params() }, Box::new(Gaussian));
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

        let mut booster = Booster::new(test_params(), Box::new(Gaussian));
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

    fn logloss(preds: &Float64Array, labels: &[f64]) -> f64 {
        preds.values().iter().zip(labels).map(|(&p, &y)| {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            - (y * p.ln() + (1.0 - y) * (1.0 - p).ln())
        }).sum::<f64>() / labels.len() as f64
    }

    #[test]
    fn test_fit_classification_reduces_logloss() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| if xi > 0.5 { 1.0 } else { 0.0 }).collect();
        let (dataset, batch) = make_dataset(x, y.clone());

        let initial_p = y.iter().sum::<f64>() / n as f64;
        let initial_logloss = logloss(&Float64Array::from(vec![initial_p; n]), &y);

        let mut booster = Booster::new(test_params(), Box::new(Logistic));
        booster.fit(&dataset);
        let preds = booster.predict(&batch);

        let final_logloss = logloss(&preds, &y);
        assert!(final_logloss < initial_logloss * 0.1);

        // also check accuracy as a sanity check
        let correct = preds.values().iter().zip(&y)
            .filter(|&(&p, &yi)| (p > 0.5) == (yi > 0.5))
            .count();
        assert!(correct as f64 / n as f64 > 0.95);
    }

    #[test]
    fn test_fit_weighted() {
        // Two points: (0, 0) and (1, 1).
        // If we weight (0, 0) heavily, the base score (mean) should be close to 0.
        // If we weight (1, 1) heavily, it should be close to 1.
        let x = vec![0.0, 1.0];
        let y = vec![0.0, 1.0];
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(x))]).unwrap();
        let labels = Float64Array::from(y);
        let params = DatasetParameters { min_data_in_bin: 1, ..DatasetParameters::default() };

        // Weight first point heavily
        let weights1 = Float64Array::from(vec![100.0, 1.0]);
        let ds1 = Dataset::from_arrow(&batch, &labels, Some(&weights1), &params);
        let mut booster1 = Booster::new(test_params(), Box::new(Gaussian));
        booster1.fit(&ds1);
        assert!(booster1.base_score < 0.1);

        // Weight second point heavily
        let weights2 = Float64Array::from(vec![1.0, 100.0]);
        let ds2 = Dataset::from_arrow(&batch, &labels, Some(&weights2), &params);
        let mut booster2 = Booster::new(test_params(), Box::new(Gaussian));
        booster2.fit(&ds2);
        assert!(booster2.base_score > 0.9);
    }
}

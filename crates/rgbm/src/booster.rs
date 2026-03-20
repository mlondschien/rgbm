use rayon::prelude::*;

use arrow::array::Float64Array;
use arrow::record_batch::RecordBatch;

use crate::dataset::Dataset;
use crate::histogram::Parameters;
use crate::objective::Objective;
use crate::tree::{Tree, TreeBuilder};

pub struct Booster {
    pub parameters: Parameters,
    pub objective: Box<dyn Objective>,
    pub trees: Vec<Tree>,
    pub base_score: f64,
    pub pool: rayon::ThreadPool,
}

impl Booster {
    pub fn new(parameters: Parameters, objective: Box<dyn Objective>) -> Self {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parameters.njobs)
            .build()
            .expect("failed to build thread pool");

        Self {
            parameters,
            objective,
            trees: Vec::new(),
            base_score: 0.0,
            pool,
        }
    }

    pub fn fit(&mut self, dataset: &Dataset) {
        let labels = dataset.labels.values();
        self.base_score = self.objective.initial_score(labels);

        let mut scores = vec![self.base_score; dataset.num_rows];
        let mut gradients = vec![0.0; dataset.num_rows];
        let mut hessians = vec![0.0; dataset.num_rows];

        self.trees.clear();
        let mut builder = TreeBuilder::new(&self.parameters, &self.pool);

        // compute gradients / hessians in parallel. Overkill for MSE or binary
        // log-loss, but maybe worth it for more complex objectives (probit?).
        for _ in 0..self.parameters.num_iterations {
            self.pool.install(|| {
                gradients.par_iter_mut()
                    .zip(hessians.par_iter_mut())
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .for_each(|(((g, h), &label), &score)| {
                        *g = self.objective.gradient(label, score);
                        *h = self.objective.hessian(label, score);
                    });
            });

            let (tree, leaf_indices) = builder.fit(dataset, &gradients, &hessians);

            for (score, &leaf_idx) in scores.iter_mut().zip(&leaf_indices) {
                *score += tree.nodes[leaf_idx as usize].value;
            }

            self.trees.push(tree);
        }
    }

    pub fn predict(&self, batch: &RecordBatch) -> Float64Array {
        let num_rows = batch.num_rows();
        let mut scores = vec![self.base_score; num_rows];

        for tree in &self.trees {
            for (score, tree_score) in scores.iter_mut().zip(tree.predict(batch)) {
                *score += tree_score;
            }
        }

        Float64Array::from_iter(scores.iter().map(|&s| self.objective.prediction(s)))
    }
}

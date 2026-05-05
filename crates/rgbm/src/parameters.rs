// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

#[derive(Clone, Debug)]
pub struct BoosterParameters {
    pub num_iterations: usize,
    pub learning_rate: f64,

    // Tree/leaf level
    pub max_depth: usize,
    pub max_leaves: usize,
    pub leaf_wise: bool,
    pub min_sum_hessian_in_leaf: f64,
    pub min_gain_to_split: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub n_jobs: isize,
}

impl Default for BoosterParameters {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            learning_rate: 0.1,
            max_depth: 6,
            max_leaves: 31,
            leaf_wise: true,
            min_sum_hessian_in_leaf: 1e-3,
            min_gain_to_split: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            n_jobs: -1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DatasetParameters {
    pub max_bin: usize,
    pub min_data_in_bin: usize,
    pub n_jobs: isize,
}

impl Default for DatasetParameters {
    fn default() -> Self {
        Self {
            max_bin: 255,
            min_data_in_bin: 3,
            n_jobs: -1,
        }
    }
}

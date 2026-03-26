#[derive(Clone, Debug)]
pub struct Parameters {
    pub num_iterations: usize,
    pub learning_rate: f64,

    // Dataset level
    pub max_bin: usize,
    pub min_data_in_bin: usize,

    // Tree/leaf level
    pub max_depth: usize,
    pub max_leaves: usize,
    pub leaf_wise: bool,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            num_iterations: 100,
            learning_rate: 0.1,
            max_bin: 255,
            min_data_in_bin: 3,
            max_depth: 6,
            max_leaves: 31,
            leaf_wise: true,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
        }
    }
}

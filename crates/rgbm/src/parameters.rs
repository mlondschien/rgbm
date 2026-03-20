#[derive(Clone, Debug)]
pub struct Parameters {
    // Booster level
    pub njobs: usize,
    pub num_iterations: usize,
    pub learning_rate: f64,

    // Tree/leaf level
    pub max_depth: usize,
    pub min_data_in_leaf: usize,
    pub min_sum_hessian_in_leaf: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            njobs: 1,
            num_iterations: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
        }
    }
}

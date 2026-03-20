use rayon::prelude::*;

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::{Float64Type, UInt32Type};

use crate::dataset::{Dataset, FeatureBinner};
use crate::histogram::{Histogram, HistogramBin, Scratch, SplitInfo, Threshold};
use crate::parameters::Parameters;
use crate::utils::{calculate_score, calculate_weight};

#[derive(Clone)]
pub enum FinalThreshold {
    Numeric(f64), 
    Categorical(Vec<bool>),
}

#[derive(Clone)]
pub struct TreeNode {
    pub is_leaf: bool,
    pub left_child: usize,
    pub right_child: usize,
    pub split_feature: usize,
    pub threshold: FinalThreshold,
    pub missing_goes_left: bool,
    pub value: f64,
}

pub struct Tree {
    pub nodes: Vec<TreeNode>,
}

impl Tree {
    /// Zero-allocation, lock-free evaluation of a single row.
    /// Columns are pre-extracted in `Booster::predict` to avoid repeated downcasting.
    #[inline(always)]
    pub fn predict_row(
        &self,
        row: usize,
        numeric_columns: &[Option<&PrimitiveArray<Float64Type>>],
        categorical_columns: &[Option<&PrimitiveArray<UInt32Type>>],
    ) -> f64 {
        let mut idx = 0;
        loop {
            let node = &self.nodes[idx];
            if node.is_leaf { return node.value; }

            let goes_left = match &node.threshold {
                FinalThreshold::Numeric(t) => {
                    let col = numeric_columns[node.split_feature].unwrap();
                    if col.is_null(row) {
                        node.missing_goes_left
                    } else {
                        let val = col.value(row);
                        if val.is_nan() { node.missing_goes_left } else { val <= *t }
                    }
                }
                FinalThreshold::Categorical(gl) => {
                    let col = categorical_columns[node.split_feature].unwrap();
                    if col.is_null(row) {
                        node.missing_goes_left
                    } else {
                        let cat_idx = col.value(row) as usize;
                        if cat_idx < gl.len() { gl[cat_idx] } else { node.missing_goes_left }
                    }
                }
            };

            idx = if goes_left { node.left_child } else { node.right_child };
        }
    }
}

pub struct TreeBuilder<'a> {
    pub parameters: &'a Parameters,
    pub nodes: Vec<TreeNode>,
    pool: &'a rayon::ThreadPool,
}

impl<'a> TreeBuilder<'a> {
    pub fn new(parameters: &'a Parameters, pool: &'a rayon::ThreadPool) -> Self {
        Self {
            parameters,
            pool,
            nodes: Vec::with_capacity((1 << (parameters.max_depth + 1)) - 1),
        }
    }

    pub fn fit(&mut self, dataset: &Dataset, gradients: &[f64], hessians: &[f64]) -> (Tree, Vec<u32>) {
        self.nodes.clear();
        let mut indices: Vec<u32> = (0..dataset.num_rows as u32).collect();
        let mut leaf_indices = vec![0u32; dataset.num_rows];

        let total_gradient: f64 = gradients.iter().sum();
        let total_hessian: f64 = hessians.iter().sum();
        let total_score = calculate_score(total_gradient, total_hessian, self.parameters.lambda_l1, self.parameters.lambda_l2);

        let pool = self.pool;
        let histograms: Vec<Histogram> = pool.install(|| {
            (0..dataset.num_features)
                .into_par_iter()
                .map(|f| Histogram::build(&dataset.binned_features[f], gradients, hessians, &indices, dataset.feature_binners[f].num_bins()))
                .collect()
        });

        self.build_node(pool, dataset, &mut indices, gradients, hessians, total_gradient, total_hessian, total_score, histograms, 0, &mut leaf_indices);

        (Tree { nodes: self.nodes.clone() }, leaf_indices)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node(
        &mut self,
        pool: &rayon::ThreadPool,
        dataset: &Dataset,
        indices: &mut [u32],
        gradients: &[f64],
        hessians: &[f64],
        total_gradient: f64,
        total_hessian: f64,
        total_score: f64,
        histograms: Vec<Histogram>,
        depth: usize,
        leaf_indices: &mut [u32],
    ) -> usize {
        let p = self.parameters;

        // possibly stop early
        if depth >= p.max_depth || indices.len() < p.min_data_in_leaf * 2 || total_hessian < p.min_sum_hessian_in_leaf * 2.0 {
            let leaf_idx = self.add_leaf(calculate_weight(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2));
            for &row in indices.iter() { leaf_indices[row as usize] = leaf_idx as u32; }
            return leaf_idx;
        }

        let best_split = pool.install(|| {
            (0..dataset.num_features)
                .into_par_iter()
                .map_init(
                    || Scratch::new(dataset.max_bins),
                    |scratch, f| {
                        let split = match &dataset.feature_binners[f] {
                            FeatureBinner::Numeric(_) => histograms[f].find_best_numeric_split(
                                total_gradient, total_hessian, indices.len() as u32, total_score, p, scratch,
                            ),
                            FeatureBinner::Categorical(_) => histograms[f].find_best_categorical_split(
                                total_gradient, total_hessian, indices.len() as u32, total_score, p, scratch,
                            ),
                        };
                        split.map(|s| (f, s))
                    },
                )
                .filter_map(|x| x)
                .max_by(|a, b| a.1.gain.partial_cmp(&b.1.gain).unwrap())
        });

        let (best_f, split) = match best_split {
            Some(res) => res,
            None => {
                let leaf_idx = self.add_leaf(calculate_weight(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2));
                for &row in indices.iter() { leaf_indices[row as usize] = leaf_idx as u32; }
                return leaf_idx;
            }
        };

        let split_idx = self.partition_indices(dataset, indices, best_f, &split);
        let (left_indices, right_indices) = indices.split_at_mut(split_idx);

        // Build histograms for the smaller child. Compute the larger child's histogram
        // by subtracting the smaller from the parent.
        let left_is_smaller = left_indices.len() < right_indices.len();
        let smaller_idx: &[u32] = if left_is_smaller { left_indices } else { right_indices };
        let (left_hists, right_hists): (Vec<Histogram>, Vec<Histogram>) = {
            pool.install(|| {
                (0..dataset.num_features)
                    .into_par_iter()
                    .map(|f| {
                        let small_hist = Histogram::build(&dataset.binned_features[f], gradients, hessians, smaller_idx, dataset.feature_binners[f].num_bins());
                        let mut large_hist = Histogram { bins: vec![HistogramBin::default(); small_hist.bins.len()] };
                        large_hist.subtract(&histograms[f], &small_hist);
                        if left_is_smaller { (small_hist, large_hist) } else { (large_hist, small_hist) }
                    })
                    .unzip()
            })
        };

        // reserve a slot for the node
        let my_idx = self.nodes.len();
        self.nodes.push(TreeNode {
            is_leaf: false, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false, value: 0.0,
        });

        let left_child = self.build_node(pool, dataset, left_indices, gradients, hessians, split.left_gradient, split.left_hessian, split.left_score, left_hists, depth + 1, leaf_indices);
        let right_child = self.build_node(pool, dataset, right_indices, gradients, hessians, split.right_gradient, split.right_hessian, split.right_score, right_hists, depth + 1, leaf_indices);

        let threshold = match &split.threshold {
            Threshold::Numeric(bin) => {
                let bound = match &dataset.feature_binners[best_f] {
                    FeatureBinner::Numeric(b) => b.upper_bounds[*bin as usize],
                    FeatureBinner::Categorical(_) => panic!("numeric split on categorical feature"),
                };
                FinalThreshold::Numeric(bound)
            }
            Threshold::Categorical(goes_left) => FinalThreshold::Categorical(goes_left.clone()),
        };
        self.nodes[my_idx] = TreeNode {
            is_leaf: false, left_child, right_child, split_feature: best_f,
            threshold, missing_goes_left: split.missing_goes_left, value: 0.0,
        };

        my_idx
    }

    fn add_leaf(&mut self, value: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false,
            value: value * self.parameters.learning_rate,
        });
        idx
    }

    fn partition_indices(&self, dataset: &Dataset, indices: &mut [u32], feature: usize, split: &SplitInfo) -> usize {
        let feature_col = &dataset.binned_features[feature];
        let sentinel = dataset.feature_binners[feature].num_bins() as u16;

        let goes_left = |row: u32| {
            let bin = feature_col[row as usize];
            if bin == sentinel {
                split.missing_goes_left
            } else {
                match &split.threshold {
                    Threshold::Numeric(t) => bin <= *t as u16,
                    Threshold::Categorical(goes_left) => goes_left[bin as usize],
                }
            }
        };

        let mut lo = 0;
        let mut hi = indices.len();
        while lo < hi {
            if goes_left(indices[lo]) { lo += 1; } else { hi -= 1; indices.swap(lo, hi); }
        }
        lo
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt32Array};

    fn leaf(value: f64) -> TreeNode {
        TreeNode { is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false, value }
    }

    fn numeric_split(feature: usize, threshold: f64, left_child: usize, right_child: usize, missing_goes_left: bool) -> TreeNode {
        TreeNode { is_leaf: false, left_child, right_child, split_feature: feature,
            threshold: FinalThreshold::Numeric(threshold), missing_goes_left, value: 0.0 }
    }

    fn predict(tree: &Tree, numeric: &[Option<Float64Array>], categorical: &[Option<UInt32Array>], row: usize) -> f64 {
        let nc: Vec<Option<&PrimitiveArray<Float64Type>>> = numeric.iter().map(|a| a.as_ref().map(|a| a as _)).collect();
        let cc: Vec<Option<&PrimitiveArray<UInt32Type>>> = categorical.iter().map(|a| a.as_ref().map(|a| a as _)).collect();
        tree.predict_row(row, &nc, &cc)
    }

    #[test]
    fn test_predict_numeric_stump() {
        // feature 0 <= 0.5 → -1.0, else → 1.0
        let tree = Tree { nodes: vec![
            numeric_split(0, 0.5, 1, 2, false),
            leaf(-1.0),
            leaf(1.0),
        ]};
        let col = vec![Some(Float64Array::from(vec![0.0, 1.0]))];
        let cat: Vec<Option<UInt32Array>> = vec![None];
        assert_eq!(predict(&tree, &col, &cat, 0), -1.0);
        assert_eq!(predict(&tree, &col, &cat, 1),  1.0);
    }

    #[test]
    fn test_predict_missing_goes_left() {
        let tree = Tree { nodes: vec![
            numeric_split(0, 0.5, 1, 2, true),   // missing → left
            leaf(-1.0),
            leaf(1.0),
        ]};
        // row 0: null → missing_goes_left → -1.0
        let col = vec![Some(Float64Array::from(vec![None as Option<f64>]))];
        let cat: Vec<Option<UInt32Array>> = vec![None];
        assert_eq!(predict(&tree, &col, &cat, 0), -1.0);
    }

    #[test]
    fn test_predict_categorical() {
        // categories 0 and 2 go left, category 1 goes right
        let tree = Tree { nodes: vec![
            TreeNode { is_leaf: false, left_child: 1, right_child: 2, split_feature: 0,
                threshold: FinalThreshold::Categorical(vec![true, false, true]),
                missing_goes_left: false, value: 0.0 },
            leaf(-1.0),
            leaf(1.0),
        ]};
        let num: Vec<Option<Float64Array>> = vec![None];
        let keys = UInt32Array::from(vec![0u32, 1, 2]);
        let cat = vec![Some(keys)];
        assert_eq!(predict(&tree, &num, &cat, 0), -1.0); // cat 0 → left
        assert_eq!(predict(&tree, &num, &cat, 1),  1.0); // cat 1 → right
        assert_eq!(predict(&tree, &num, &cat, 2), -1.0); // cat 2 → left
    }

    #[test]
    fn test_predict_deep_tree() {
        // feature 0 splits at 0.5; left child splits feature 1 at 0.5
        // [0,0]→-2, [0,1]→-1, [1,*]→1
        let tree = Tree { nodes: vec![
            numeric_split(0, 0.5, 1, 4, false),  // root: f0 <= 0.5 → node 1, else → node 4
            numeric_split(1, 0.5, 2, 3, false),  // node 1: f1 <= 0.5 → node 2, else → node 3
            leaf(-2.0),
            leaf(-1.0),
            leaf(1.0),
        ]};
        let f0 = Float64Array::from(vec![0.0, 0.0, 1.0]);
        let f1 = Float64Array::from(vec![0.0, 1.0, 0.0]);
        let col = vec![Some(f0), Some(f1)];
        let cat: Vec<Option<UInt32Array>> = vec![None, None];
        assert_eq!(predict(&tree, &col, &cat, 0), -2.0);
        assert_eq!(predict(&tree, &col, &cat, 1), -1.0);
        assert_eq!(predict(&tree, &col, &cat, 2),  1.0);
    }
}
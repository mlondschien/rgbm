use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::{Float64Type, UInt32Type};

use crate::dataset::{Dataset, FeatureBinner};
use crate::histogram::{Histogram, HistogramBin, SplitInfo, Threshold};
use crate::parameters::Parameters;
use crate::utils::{calculate_score, calculate_weight};

#[derive(Clone)]
pub enum FinalThreshold {
    Numeric(f64),
    Categorical(Vec<bool>),
}

#[derive(Clone)]
pub struct Node {
    pub is_leaf: bool,
    pub left_child: usize,
    pub right_child: usize,
    pub split_feature: usize,
    pub threshold: FinalThreshold,
    pub missing_goes_left: bool,
    pub value: f64,
}

struct Leaf {
    node_idx: usize,
    indices: Vec<u32>,
    histograms: Vec<Histogram>,
    best_split: Option<(usize, SplitInfo)>,
    depth: usize,
}

fn find_best_split(
    dataset: &Dataset,
    p: &Parameters,
    hists: &[Histogram],
    total_gradient: f64,
    total_hessian: f64,
    n: u32,
    parent_score: f64,
    depth: usize,
) -> Option<(usize, SplitInfo)> {
    if depth >= p.max_depth || (n as usize) < p.min_data_in_leaf * 2 || total_hessian < p.min_sum_hessian_in_leaf * 2.0 {
        return None;
    }
    (0..dataset.num_features)
        .filter_map(|f| {
            let split = match &dataset.feature_binners[f] {
                FeatureBinner::Numeric(_) => hists[f].find_best_numeric_split(total_gradient, total_hessian, n, parent_score, p),
                FeatureBinner::Categorical(_) => hists[f].find_best_categorical_split(total_gradient, total_hessian, n, parent_score, p),
            };
            split.map(|s| (f, s))
        })
        .max_by(|a, b| a.1.gain.partial_cmp(&b.1.gain).unwrap())
}

#[derive(Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub fn new(max_leaves: usize) -> Self {
        Self { nodes: Vec::with_capacity(2 * max_leaves) }
    }

    pub fn fit(&mut self, dataset: &Dataset, gradients: &[f64], hessians: &[f64], p: &Parameters) -> Vec<u32> {
        self.nodes.clear();
        let mut leaf_indices = vec![0u32; dataset.num_rows];

        let total_gradient: f64 = gradients.iter().sum();
        let total_hessian: f64 = hessians.iter().sum();
        let total_score = calculate_score(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2);

        let indices: Vec<u32> = (0..dataset.num_rows as u32).collect();
        let histograms: Vec<Histogram> = (0..dataset.num_features)
            .map(|f| Histogram::build(&dataset.binned_features[f], gradients, hessians, &indices, dataset.feature_binners[f].num_bins()))
            .collect();

        let best_split = find_best_split(dataset, p, &histograms, total_gradient, total_hessian, indices.len() as u32, total_score, 0);
        let root_node = self.add_leaf(calculate_weight(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2), p);

        let mut leaves = vec![Leaf { node_idx: root_node, indices, histograms, best_split, depth: 0 }];
        let mut num_leaves = 1;

        while num_leaves < p.max_leaves {
            // Pick the leaf with the highest gain split.
            let best_i = match leaves.iter().enumerate()
                .filter_map(|(i, l)| l.best_split.as_ref().map(|(_, s)| (i, s.gain)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                Some((i, _)) => i,
                None => break,
            };

            let Leaf { node_idx, mut indices, histograms, best_split, depth } = leaves.swap_remove(best_i);
            let (best_f, split) = best_split.unwrap();

            let split_pos = self.partition_indices(dataset, &mut indices, best_f, &split);

            // Build histograms for the smaller child; subtract for the larger.
            let left_histograms: Vec<Histogram>;
            let right_histograms: Vec<Histogram>;
            if split_pos < indices.len() - split_pos {
                left_histograms = (0..dataset.num_features)
                    .map(|f| Histogram::build(&dataset.binned_features[f], gradients, hessians, &indices[..split_pos], dataset.feature_binners[f].num_bins()))
                    .collect();
                right_histograms = (0..dataset.num_features).map(|f| {
                    let mut h = Histogram { bins: vec![HistogramBin::default(); left_histograms[f].bins.len()] };
                    h.subtract(&histograms[f], &left_histograms[f]);
                    h
                }).collect();
            } else {
                right_histograms = (0..dataset.num_features)
                    .map(|f| Histogram::build(&dataset.binned_features[f], gradients, hessians, &indices[split_pos..], dataset.feature_binners[f].num_bins()))
                    .collect();
                left_histograms = (0..dataset.num_features).map(|f| {
                    let mut h = Histogram { bins: vec![HistogramBin::default(); right_histograms[f].bins.len()] };
                    h.subtract(&histograms[f], &right_histograms[f]);
                    h
                }).collect();
            }

            let left_indices = indices[..split_pos].to_vec();
            let right_indices = indices[split_pos..].to_vec();

            let left_node = self.add_leaf(calculate_weight(split.left_gradient, split.left_hessian, p.lambda_l1, p.lambda_l2), p);
            let right_node = self.add_leaf(calculate_weight(split.right_gradient, split.right_hessian, p.lambda_l1, p.lambda_l2), p);

            let threshold = match &split.threshold {
                Threshold::Numeric(bin) => FinalThreshold::Numeric(match &dataset.feature_binners[best_f] {
                    FeatureBinner::Numeric(b) => b.upper_bounds[*bin as usize],
                    FeatureBinner::Categorical(_) => panic!("numeric split on categorical feature"),
                }),
                Threshold::Categorical(gl) => FinalThreshold::Categorical(gl.clone()),
            };
            self.nodes[node_idx] = Node {
                is_leaf: false, left_child: left_node, right_child: right_node,
                split_feature: best_f, threshold, missing_goes_left: split.missing_goes_left, value: 0.0,
            };

            let left_best_split = find_best_split(dataset, p, &left_histograms, split.left_gradient, split.left_hessian, left_indices.len() as u32, split.left_score, depth + 1);
            let right_best_split = find_best_split(dataset, p, &right_histograms, split.right_gradient, split.right_hessian, right_indices.len() as u32, split.right_score, depth + 1);

            leaves.push(Leaf { node_idx: left_node, indices: left_indices, histograms: left_histograms, best_split: left_best_split, depth: depth + 1 });
            leaves.push(Leaf { node_idx: right_node, indices: right_indices, histograms: right_histograms, best_split: right_best_split, depth: depth + 1 });
            num_leaves += 1;
        }

        for leaf in &leaves {
            for &row in &leaf.indices {
                leaf_indices[row as usize] = leaf.node_idx as u32;
            }
        }

        leaf_indices
    }

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

    fn add_leaf(&mut self, value: f64, p: &Parameters) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false,
            value: value * p.learning_rate,
        });
        idx
    }

    fn partition_indices(&self, dataset: &Dataset, indices: &mut [u32], feature: usize, split: &SplitInfo) -> usize {
        let feature_col = &dataset.binned_features[feature];
        let sentinel = dataset.feature_binners[feature].num_bins() as u8;

        let goes_left = |row: u32| {
            let bin = feature_col[row as usize];
            if bin == sentinel {
                split.missing_goes_left
            } else {
                match &split.threshold {
                    Threshold::Numeric(t) => bin <= *t as u8,
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

    fn leaf(value: f64) -> Node {
        Node { is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false, value }
    }

    fn numeric_split(feature: usize, threshold: f64, left_child: usize, right_child: usize, missing_goes_left: bool) -> Node {
        Node { is_leaf: false, left_child, right_child, split_feature: feature,
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
            Node { is_leaf: false, left_child: 1, right_child: 2, split_feature: 0,
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

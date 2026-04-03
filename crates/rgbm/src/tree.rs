use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::{Float64Type, UInt32Type};

use crate::bin::FeatureBinner;
use crate::dataset::Dataset;
use crate::histogram::{Histogram, SplitInfo, Threshold};
use crate::parameters::Parameters;
use crate::utils::{calculate_score, calculate_weight};

#[derive(Clone)]
pub enum FinalThreshold {
    Numeric(f64),
    Categorical(Vec<bool>),
}

pub struct Node {
    pub is_leaf: bool,
    pub left_child: usize,
    pub right_child: usize,
    pub split_feature: usize,
    pub threshold: FinalThreshold,
    pub missing_goes_left: bool,
    pub value: f64,
    // Training state — only meaningful for leaf nodes during Tree::fit.
    start: usize,
    len: usize,
    histograms: Vec<Histogram>,
    best_split: Option<(usize, SplitInfo)>,
    depth: usize,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            is_leaf: self.is_leaf, left_child: self.left_child, right_child: self.right_child,
            split_feature: self.split_feature, threshold: self.threshold.clone(),
            missing_goes_left: self.missing_goes_left, value: self.value,
            start: 0, len: 0, histograms: Vec::new(), best_split: None, depth: 0,
        }
    }
}

fn find_best_split(
    p: &Parameters,
    histograms: &[Histogram],
    total_gradient: f64,
    total_hessian: f64,
    parent_score: f64,
    depth: usize,
) -> Option<(usize, SplitInfo)> {
    if depth >= p.max_depth || total_hessian < p.min_sum_hessian_in_leaf * 2.0 {
        return None;
    }
    (0..histograms.len())
        .filter_map(|f| {
            histograms[f].find_best_split(total_gradient, total_hessian, parent_score, p)
                .map(|s| (f, s))
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

    pub fn fit(&mut self, dataset: &Dataset, grad_hess: &[[f64; 2]], p: &Parameters) -> Vec<u32> {
        self.nodes.clear();
        let mut leaf_indices = vec![0u32; dataset.num_rows];
        let mut left_buffer = vec![0u32; dataset.num_rows];
        let mut right_buffer = vec![0u32; dataset.num_rows];
        let mut all_indices: Vec<u32> = (0..dataset.num_rows as u32).collect();

        let total_gradient: f64 = grad_hess.iter().map(|gh| gh[0]).sum();
        let total_hessian: f64 = grad_hess.iter().map(|gh| gh[1]).sum();
        let total_score = calculate_score(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2);

        let mut histograms: Vec<Histogram> = Vec::with_capacity(dataset.num_features);
        for b in &dataset.feature_bundles {
            histograms.extend(Histogram::build(b, grad_hess, &all_indices));
        }

        let best_split = find_best_split(p, &histograms, total_gradient, total_hessian, total_score, 0);
        let root = self.add_node(calculate_weight(total_gradient, total_hessian, p.lambda_l1, p.lambda_l2), p, 0, dataset.num_rows, histograms, best_split, 0);

        let mut leaves = vec![root];
        let mut num_leaves = 1;

        while num_leaves < p.max_leaves {
            let best_i = match leaves.iter().enumerate()
                .filter_map(|(i, &ni)| self.nodes[ni].best_split.as_ref().map(|(_, s)| (i, s.gain)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            {
                Some((i, _)) => i,
                None => break,
            };

            let node_idx = leaves.swap_remove(best_i);
            let start = self.nodes[node_idx].start;
            let len = self.nodes[node_idx].len;
            let depth = self.nodes[node_idx].depth;
            let histograms = std::mem::take(&mut self.nodes[node_idx].histograms);
            let (best_f, split) = std::mem::take(&mut self.nodes[node_idx].best_split).unwrap();

            let slice = &mut all_indices[start..start + len];
            let split_pos = self.partition_indices(dataset, slice, best_f, &split, &mut left_buffer, &mut right_buffer);

            let left_start = start;
            let left_len = split_pos;
            let right_start = start + split_pos;
            let right_len = len - split_pos;

            // Build histograms for the smaller child; subtract for the larger.
            let mut left_histograms: Vec<Histogram>;
            let mut right_histograms: Vec<Histogram>;
            if left_len < right_len {
                left_histograms = Vec::with_capacity(dataset.num_features);
                for b in &dataset.feature_bundles {
                    left_histograms.extend(Histogram::build(b, grad_hess, &all_indices[left_start..left_start + left_len]));
                }
                right_histograms = (0..dataset.num_features).map(|f| {
                    let mut h = Histogram::zeros(left_histograms[f].bins.len(), left_histograms[f].is_categorical);
                    h.subtract(&histograms[f], &left_histograms[f]);
                    h
                }).collect();
            } else {
                right_histograms = Vec::with_capacity(dataset.num_features);
                for b in &dataset.feature_bundles {
                    right_histograms.extend(Histogram::build(b, grad_hess, &all_indices[right_start..right_start + right_len]));
                }
                left_histograms = (0..dataset.num_features).map(|f| {
                    let mut h = Histogram::zeros(right_histograms[f].bins.len(), right_histograms[f].is_categorical);
                    h.subtract(&histograms[f], &right_histograms[f]);
                    h
                }).collect();
            }

            let left_best_split = find_best_split(p, &left_histograms, split.left_gradient, split.left_hessian, split.left_score, depth + 1);
            let right_best_split = find_best_split(p, &right_histograms, split.right_gradient, split.right_hessian, split.right_score, depth + 1);

            let left_node = self.add_node(calculate_weight(split.left_gradient, split.left_hessian, p.lambda_l1, p.lambda_l2), p, left_start, left_len, left_histograms, left_best_split, depth + 1);
            let right_node = self.add_node(calculate_weight(split.right_gradient, split.right_hessian, p.lambda_l1, p.lambda_l2), p, right_start, right_len, right_histograms, right_best_split, depth + 1);

            let threshold = match &split.threshold {
                Threshold::Numeric(bin) => FinalThreshold::Numeric(match &dataset.feature_binners[best_f] {
                    FeatureBinner::Numerical(upper_bounds) => upper_bounds[*bin as usize],
                    FeatureBinner::Categorical(_) => panic!("numeric split on categorical feature"),
                }),
                Threshold::Categorical(gl) => FinalThreshold::Categorical(gl.clone()),
            };
            self.nodes[node_idx].is_leaf = false;
            self.nodes[node_idx].left_child = left_node;
            self.nodes[node_idx].right_child = right_node;
            self.nodes[node_idx].split_feature = best_f;
            self.nodes[node_idx].threshold = threshold;
            self.nodes[node_idx].missing_goes_left = split.missing_goes_left;
            self.nodes[node_idx].value = 0.0;

            leaves.push(left_node);
            leaves.push(right_node);
            num_leaves += 1;
        }

        for &ni in &leaves {
            let node = &self.nodes[ni];
            for &row in &all_indices[node.start..node.start + node.len] {
                leaf_indices[row as usize] = ni as u32;
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

    fn add_node(&mut self, value: f64, p: &Parameters, start: usize, len: usize, histograms: Vec<Histogram>, best_split: Option<(usize, SplitInfo)>, depth: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false,
            value: value * p.learning_rate,
            start, len, histograms, best_split, depth,
        });
        idx
    }

    fn partition_indices(&self, dataset: &Dataset, indices: &mut [u32], feature: usize, split: &SplitInfo, left_buffer: &mut [u32], right_buffer: &mut [u32]) -> usize {
        let bundle_idx = feature / 4;
        let shift = (feature % 4) * 8;
        let packed_bins = &dataset.feature_bundles[bundle_idx].packed_bins;
        let sentinel = dataset.feature_binners[feature].num_bins() as u8;
        let missing = split.missing_goes_left as usize;
        let mut left_count = 0usize;
        let mut right_count = 0usize;

        match &split.threshold {
            Threshold::Numeric(t) => {
                let t = *t as u8;
                for &row in indices.iter() {
                    let bin = ((packed_bins[row as usize] >> shift) & 0xFF) as u8;
                    let goes_left = if bin == sentinel { missing } else { (bin <= t) as usize };
                    left_buffer[left_count] = row;
                    right_buffer[right_count] = row;
                    left_count += goes_left;
                    right_count += 1 - goes_left;
                }
            }
            Threshold::Categorical(cats) => {
                for &row in indices.iter() {
                    let bin = ((packed_bins[row as usize] >> shift) & 0xFF) as usize;
                    let goes_left = if bin == sentinel as usize { missing } else { cats[bin] as usize };
                    left_buffer[left_count] = row;
                    right_buffer[right_count] = row;
                    left_count += goes_left;
                    right_count += 1 - goes_left;
                }
            }
        }

        indices[..left_count].copy_from_slice(&left_buffer[..left_count]);
        indices[left_count..].copy_from_slice(&right_buffer[..right_count]);
        left_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float64Array, UInt32Array};

    fn leaf(value: f64) -> Node {
        Node { is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false, value,
            start: 0, len: 0, histograms: Vec::new(), best_split: None, depth: 0 }
    }

    fn numeric_split(feature: usize, threshold: f64, left_child: usize, right_child: usize, missing_goes_left: bool) -> Node {
        Node { is_leaf: false, left_child, right_child, split_feature: feature,
            threshold: FinalThreshold::Numeric(threshold), missing_goes_left, value: 0.0,
            start: 0, len: 0, histograms: Vec::new(), best_split: None, depth: 0 }
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
                missing_goes_left: false, value: 0.0,
                start: 0, len: 0, histograms: Vec::new(), best_split: None, depth: 0 },
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

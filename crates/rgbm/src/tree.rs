use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::Float64Type;

use crate::bin::FeatureBinner;
use crate::dataset::Dataset;
use crate::histogram::{Histogram, SplitInfo, Threshold};
use crate::parameters::Parameters;
use crate::histogram::calculate_score;

#[derive(Clone)]
pub enum FinalThreshold {
    Numeric(f64),
    Categorical(Vec<bool>),
}

impl FinalThreshold {
    pub fn from_threshold(threshold: &Threshold, binner: &FeatureBinner) -> Self {
        match threshold {
            Threshold::Numeric(bin) => FinalThreshold::Numeric(match binner {
                FeatureBinner::Numerical(upper_bounds) => upper_bounds[*bin as usize],
                FeatureBinner::Categorical(_) => panic!("numeric split on categorical feature"),
            }),
            Threshold::Categorical(gl) => FinalThreshold::Categorical(gl.clone()),
        }
    }
}

/// Node in the decision tree, stored after training.
// why not use an enum: Leaf/Internal? I did this initially, but was told off by an LLM.
// Apparently this way storing both leaf and internal fields together with an is_leaf
// flag makes the predict much faster.
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

/// Leaf node during training. In leaf-first growth, we keep track of active leaves and
/// select the next leaf to split based on the best split gain.
pub struct ActiveLeaf {
    leaf_index: usize,
    start: usize,
    len: usize,
    depth: usize,
    histograms: Vec<Histogram>,
    best_split: SplitInfo,
}



fn find_best_split(
    p: &Parameters,
    histograms: &[Histogram],
    total_gradient: f64,
    total_hessian: f64,
    parent_score: f64,
    depth: usize,
) -> Option<SplitInfo> {
    if depth >= p.max_depth || total_hessian < p.min_sum_hessian_in_leaf * 2.0 {
        return None;
    }

    let mut best_split: Option<SplitInfo> = None;

    for f in 0..histograms.len() {
        if let Some(mut split) = histograms[f].find_best_split(total_gradient, total_hessian, parent_score, p) {
            split.feature_index = f;

            if best_split.as_ref().map_or(true, |best| split.gain > best.gain) {
                best_split = Some(split);
            }
        }
    }

    best_split
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
        let mut active_leafs: Vec<ActiveLeaf> = Vec::new();

        let root_histograms = Self::build_histograms(dataset, grad_hess, &all_indices);
        self.push_leaf(&mut active_leafs, &mut leaf_indices, &all_indices, root_histograms, 0, dataset.num_rows, 0, p);
        let mut num_leaves = 1;

        while num_leaves < p.max_leaves && !active_leafs.is_empty() {
            let (idx, _) = active_leafs.iter().enumerate().max_by(|(_, a), (_, b)| {
                a.best_split.gain.total_cmp(&b.best_split.gain)
            }).unwrap();
            let leaf = active_leafs.swap_remove(idx);

            let split_position = self.partition_indices(
                dataset, &mut all_indices[leaf.start..leaf.start + leaf.len], &leaf.best_split, &mut left_buffer, &mut right_buffer);

            let left_start = leaf.start;
            let left_len = split_position;
            let right_start = leaf.start + split_position;
            let right_len = leaf.len - split_position;

            // Build the smaller child directly, derive the larger by subtracting
            let (left_histograms, right_histograms);
            if left_len < right_len {
                left_histograms = Self::build_histograms(dataset, grad_hess, &all_indices[left_start..left_start + left_len]);
                right_histograms = Self::subtract_histograms(dataset.num_features, &leaf.histograms, &left_histograms);
            } else {
                right_histograms = Self::build_histograms(dataset, grad_hess, &all_indices[right_start..right_start + right_len]);
                left_histograms = Self::subtract_histograms(dataset.num_features, &leaf.histograms, &right_histograms);
            }

            let left_node_idx = self.push_leaf(&mut active_leafs, &mut leaf_indices, &all_indices, left_histograms, left_start, left_len, leaf.depth + 1, p);
            let right_node_idx = self.push_leaf(&mut active_leafs, &mut leaf_indices, &all_indices, right_histograms, right_start, right_len, leaf.depth + 1, p);

            let parent = &mut self.nodes[leaf.leaf_index];
            parent.is_leaf = false;
            parent.left_child = left_node_idx;
            parent.right_child = right_node_idx;
            parent.split_feature = leaf.best_split.feature_index;
            parent.missing_goes_left = leaf.best_split.missing_goes_left;
            parent.threshold = FinalThreshold::from_threshold(
                &leaf.best_split.threshold,
                &dataset.feature_binners[leaf.best_split.feature_index],
            );
            num_leaves += 1;
        }

        for leaf in active_leafs {
            for &row in &all_indices[leaf.start..leaf.start + leaf.len] {
                leaf_indices[row as usize] = leaf.leaf_index as u32;
            }
        }

        leaf_indices
    }

    /// Zero-allocation, lock-free evaluation of a single row. For speed.
    #[inline(always)]
    pub fn predict_row(
        &self,
        row: usize,
        numeric_columns: &[Option<&PrimitiveArray<Float64Type>>],
        categorical_columns: &[Option<Vec<u8>>],
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
                    let col = categorical_columns[node.split_feature].as_ref().unwrap();
                    let bin = col[row] as usize;
                    if bin < gl.len() { gl[bin] } else { node.missing_goes_left }
                }
            };

            idx = if goes_left { node.left_child } else { node.right_child };
        }
    }

    fn push_leaf(
        &mut self,
        active_leafs: &mut Vec<ActiveLeaf>,
        leaf_indices: &mut Vec<u32>,
        all_indices: &[u32],
        histograms: Vec<Histogram>,
        start: usize,
        len: usize,
        depth: usize,
        p: &Parameters,
    ) -> usize {
        // todo: This will panic if there's no features.
        let (gradient, hessian) = histograms[0].bins.iter()
            .fold((0.0, 0.0), |(g, h), b| (g + b.sum_gradients, h + b.sum_hessians));
        let score = calculate_score(gradient, hessian, p.lambda_l1, p.lambda_l2);
        let value =calculate_value(gradient, hessian, p.lambda_l1, p.lambda_l2) * p.learning_rate;
        let node_idx = self.nodes.len();
        self.nodes.push(Node {
            is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false,
            value: value,
        });
        match find_best_split(p, &histograms, gradient, hessian, score, depth) {
            Some(best_split) => active_leafs.push(ActiveLeaf { leaf_index: node_idx, start, len, depth, histograms, best_split }),
            None => {
                for &row in &all_indices[start..start + len] {
                    leaf_indices[row as usize] = node_idx as u32;
                }
            }
        }
        node_idx
    }

    fn build_histograms(dataset: &Dataset, grad_hess: &[[f64; 2]], indices: &[u32]) -> Vec<Histogram> {
        let mut hists = Vec::with_capacity(dataset.num_features);
        for b in &dataset.feature_bundles {
            hists.extend(Histogram::build(b, grad_hess, indices));
        }
        hists
    }

    fn subtract_histograms(num_features: usize, parent: &[Histogram], child: &[Histogram]) -> Vec<Histogram> {
        (0..num_features).map(|f| {
            let mut h = Histogram::zeros(child[f].bins.len(), child[f].is_categorical);
            h.subtract(&parent[f], &child[f]);
            h
        }).collect()
    }

    fn partition_indices(&self, dataset: &Dataset, indices: &mut [u32], split: &SplitInfo, left_buffer: &mut [u32], right_buffer: &mut [u32]) -> usize {
        let bundle_idx = split.feature_index / 4;
        let shift = (split.feature_index % 4) * 8;
        let packed_bins = &dataset.feature_bundles[bundle_idx].packed_bins;
        
        let sentinel = (dataset.feature_binners[split.feature_index].num_bins() - 1) as u8;
        
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

#[inline(always)]
pub fn calculate_value(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    let d = (g.abs() - l1).max(0.0);
    -g.signum() * d / (h + l2)
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn leaf(value: f64) -> Node {
        Node { is_leaf: true, left_child: 0, right_child: 0, split_feature: 0,
            threshold: FinalThreshold::Numeric(0.0), missing_goes_left: false, value }
    }

    fn numeric_split(feature: usize, threshold: f64, left_child: usize, right_child: usize, missing_goes_left: bool) -> Node {
        Node { is_leaf: false, left_child, right_child, split_feature: feature,
            threshold: FinalThreshold::Numeric(threshold), missing_goes_left, value: 0.0 }
    }

    fn predict(tree: &Tree, numeric: &[Option<Float64Array>], categorical: &[Option<Vec<u8>>], row: usize) -> f64 {
        let nc: Vec<Option<&PrimitiveArray<Float64Type>>> = numeric.iter().map(|a| a.as_ref().map(|a| a as _)).collect();
        tree.predict_row(row, &nc, categorical)
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
        let cat: Vec<Option<Vec<u8>>> = vec![None];
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
        let cat: Vec<Option<Vec<u8>>> = vec![None];
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
        let cat = vec![Some(vec![0u8, 1, 2])]; // bin indices for rows 0, 1, 2
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
        let cat: Vec<Option<Vec<u8>>> = vec![None, None];
        assert_eq!(predict(&tree, &col, &cat, 0), -2.0);
        assert_eq!(predict(&tree, &col, &cat, 1), -1.0);
        assert_eq!(predict(&tree, &col, &cat, 2),  1.0);
    }
}

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::Float64Type;

use crate::bin::FeatureBinner;
use crate::dataset::Dataset;
use crate::histogram::{Histograms, SplitInfo, Threshold};
use crate::parameters::BoosterParameters;
use crate::histogram::calculate_score;
use crate::utils::prefetch;

/// Reusable scratch buffers for tree fitting, owned by the Booster and passed in each
/// iteration to avoid repeated allocation of O(num_rows) memory.
pub struct TreeWorkspace {
    pub leaf_indices: Vec<u32>,
    left_buffer: Vec<u32>,
    right_buffer: Vec<u32>,
    all_indices: Vec<u32>,
    ordered_gh: Vec<[f32; 2]>,
}

impl TreeWorkspace {
    pub fn new(num_rows: usize) -> Self {
        Self {
            leaf_indices: vec![0u32; num_rows],
            left_buffer: vec![0u32; num_rows],
            right_buffer: vec![0u32; num_rows],
            all_indices: (0..num_rows as u32).collect(),
            ordered_gh: vec![[0.0f32; 2]; num_rows],
        }
    }
}

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
#[derive(Clone)]
#[repr(u8)]
pub enum Node {
    Leaf {
        value: f64,
    },
    Internal {
        left_child: u32,
        right_child: u32,
        split_feature: u32,
        missing_goes_left: bool,
        threshold: FinalThreshold,
    },
}

impl Node {
    #[inline(always)]
    pub fn value(&self) -> f64 {
        match self {
            Node::Leaf { value } => *value,
            _ => panic!("called value() on an internal node"),
        }
    }
}

/// Leaf node during training. In leaf-first growth, we keep track of active leaves and
/// select the next leaf to split based on the best split gain.
pub struct ActiveLeaf {
    leaf_index: usize,
    start: usize,
    len: usize,
    depth: usize,
    histograms: Histograms,
    best_split: SplitInfo,
}

#[derive(Clone)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub fn new(max_leaves: usize) -> Self {
        Self { nodes: Vec::with_capacity(2 * max_leaves) }
    }

    pub fn fit(&mut self, dataset: &Dataset, grad_hess: &[[f32; 2]], p: &BoosterParameters, pool: Option<&rayon::ThreadPool>, workspace: &mut TreeWorkspace) {
        self.nodes.clear();

        for (i, x) in workspace.all_indices.iter_mut().enumerate() { *x = i as u32; }

        let mut active_leafs: Vec<ActiveLeaf> = Vec::new();

        let root_histograms = Histograms::build(&dataset.feature_bundles, grad_hess, &workspace.all_indices, pool);

        self.push_leaf(&mut active_leafs, &mut workspace.leaf_indices, &workspace.all_indices, root_histograms, 0, dataset.num_rows, 0, p, pool);
        let mut num_leaves = 1;

        while num_leaves < p.max_leaves && !active_leafs.is_empty() {
            let (idx, _) = active_leafs.iter().enumerate().max_by(|(_, a), (_, b)| {
                a.best_split.gain.total_cmp(&b.best_split.gain)
            }).unwrap();
            let leaf = active_leafs.swap_remove(idx);

            let split_position = self.partition_indices(
                dataset, &mut workspace.all_indices[leaf.start..leaf.start + leaf.len],
                &leaf.best_split, &mut workspace.left_buffer, &mut workspace.right_buffer);

            let left_start = leaf.start;
            let left_len = split_position;
            let right_start = leaf.start + split_position;
            let right_len = leaf.len - split_position;

            // Build the smaller child directly, derive the larger by subtracting
            let (mut left_histograms, mut right_histograms);
            if left_len < right_len {
                let left_indices = &workspace.all_indices[left_start..left_start + left_len];
                let ordered_grad_hess = &mut workspace.ordered_gh[..left_len];
                gather_gradients(left_indices, grad_hess, ordered_grad_hess);
                left_histograms = Histograms::build(&dataset.feature_bundles, ordered_grad_hess, left_indices, pool);
                right_histograms = leaf.histograms;
                right_histograms.subtract(&left_histograms);
            } else {
                let right_indices = &workspace.all_indices[right_start..right_start + right_len];
                let ordered_grad_hess = &mut workspace.ordered_gh[..right_len];
                gather_gradients(right_indices, grad_hess, ordered_grad_hess);
                right_histograms = Histograms::build(&dataset.feature_bundles, ordered_grad_hess, right_indices, pool);
                left_histograms = leaf.histograms;
                left_histograms.subtract(&right_histograms);
            }

            let left_node_idx = self.push_leaf(&mut active_leafs, &mut workspace.leaf_indices, &workspace.all_indices, left_histograms, left_start, left_len, leaf.depth + 1, p, pool);
            let right_node_idx = self.push_leaf(&mut active_leafs, &mut workspace.leaf_indices, &workspace.all_indices, right_histograms, right_start, right_len, leaf.depth + 1, p, pool);

            self.nodes[leaf.leaf_index] = Node::Internal {
                left_child: left_node_idx as u32,
                right_child: right_node_idx as u32,
                split_feature: leaf.best_split.feature_index as u32,
                missing_goes_left: leaf.best_split.missing_goes_left,
                threshold: FinalThreshold::from_threshold(
                    &leaf.best_split.threshold,
                    &dataset.feature_binners[leaf.best_split.feature_index],
                ),
            };
            num_leaves += 1;
        }

        for leaf in active_leafs {
            for &row in &workspace.all_indices[leaf.start..leaf.start + leaf.len] {
                workspace.leaf_indices[row as usize] = leaf.leaf_index as u32;
            }
        }
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
            match unsafe { self.nodes.get_unchecked(idx) } {
                Node::Leaf { value } => return *value,
                Node::Internal {
                    left_child,
                    right_child,
                    split_feature,
                    missing_goes_left,
                    threshold,
                } => {
                    let goes_left = match threshold {
                        FinalThreshold::Numeric(t) => {
                            let col = numeric_columns[*split_feature as usize].unwrap();
                            if col.is_null(row) {
                                *missing_goes_left
                            } else {
                                let val = col.value(row);
                                if val.is_nan() { *missing_goes_left } else { val <= *t }
                            }
                        }
                        FinalThreshold::Categorical(gl) => {
                            let col = categorical_columns[*split_feature as usize].as_ref().unwrap();
                            let bin = col[row] as usize;
                            if bin < gl.len() { gl[bin] } else { *missing_goes_left }
                        }
                    };
                    idx = if goes_left { *left_child as usize } else { *right_child as usize };
                }
            }
        }
    }

    fn push_leaf(
        &mut self,
        active_leafs: &mut Vec<ActiveLeaf>,
        leaf_indices: &mut Vec<u32>,
        all_indices: &[u32],
        histograms: Histograms,
        start: usize,
        len: usize,
        depth: usize,
        p: &BoosterParameters,
        pool: Option<&rayon::ThreadPool>,
    ) -> usize {
        // This returns 0 if there's no features.
        let end = histograms.offsets.get(1).copied().unwrap_or(histograms.bins.len());
        let (gradient, hessian) = histograms.bins[..end].iter()
            .fold((0.0, 0.0), |(g, h), b| (g + b.sum_gradients, h + b.sum_hessians));
        let score = calculate_score(gradient, hessian, p.lambda_l1, p.lambda_l2);
        let value = calculate_value(gradient, hessian, p.lambda_l1, p.lambda_l2) * p.learning_rate;
        let node_idx = self.nodes.len();

        self.nodes.push(Node::Leaf { value });

        let best_split = if depth >= p.max_depth || hessian < p.min_sum_hessian_in_leaf * 2.0 {
            None
        } else {
            histograms.find_best_split(gradient, hessian, score, p, pool)
        };

        match best_split {
            Some(best_split) => active_leafs.push(ActiveLeaf { leaf_index: node_idx, start, len, depth, histograms, best_split }),
            None => {
                for &row in &all_indices[start..start + len] {
                    leaf_indices[row as usize] = node_idx as u32;
                }
            }
        }
        node_idx
    }

    pub fn partition_indices(
        &self,
        dataset: &Dataset,
        indices: &mut [u32],
        split: &SplitInfo,
        left_buffer: &mut [u32],
        right_buffer: &mut [u32],
    ) -> usize {
        let bundle_idx = split.feature_index / 4;
        let shift = (split.feature_index % 4) * 8;
        let packed_bins = &dataset.feature_bundles[bundle_idx].packed_bins;

        let sentinel = (dataset.feature_binners[split.feature_index].num_bins() - 1) as u8;

        let missing = split.missing_goes_left as usize;

        const PREFETCH_DIST: usize = 16;
        let n = indices.len();
        let mid = n.saturating_sub(PREFETCH_DIST);

        let mut left_count = 0usize;
        let mut right_count = 0usize;

        // Cache the raw pointer to bypass slice reference overhead
        let pb_ptr = packed_bins.as_ptr();

        match &split.threshold {
            Threshold::Numeric(t) => {
                let t = *t as u8;
                for i in 0..mid {
                    unsafe {
                        let prefetch_index = *indices.get_unchecked(i + PREFETCH_DIST) as usize;
                        prefetch(pb_ptr.add(prefetch_index  as usize));

                        let index = *indices.get_unchecked(i);
                        let bin = ((*pb_ptr.add(index as usize) >> shift) & 0xFF) as u8;
                        // branchless calculation
                        let goes_left = if bin == sentinel { missing } else { (bin <= t) as usize };

                        *left_buffer.get_unchecked_mut(left_count) = index;
                        *right_buffer.get_unchecked_mut(right_count) = index;

                        left_count += goes_left;
                        right_count += 1 - goes_left;
                    }
                }
                for i in mid..n {
                    unsafe {
                        let index = *indices.get_unchecked(i);
                        let bin = ((*pb_ptr.add(index as usize) >> shift) & 0xFF) as u8;
                        let goes_left = if bin == sentinel { missing } else { (bin <= t) as usize };

                        *left_buffer.get_unchecked_mut(left_count) = index;
                        *right_buffer.get_unchecked_mut(right_count) = index;

                        left_count += goes_left;
                        right_count += 1 - goes_left;
                    }
                }
            }
            Threshold::Categorical(cats) => {
                for i in 0..mid {
                    unsafe {
                        let ahead_row = *indices.get_unchecked(i + PREFETCH_DIST) as usize;
                        prefetch(pb_ptr.add(ahead_row));

                        let row = *indices.get_unchecked(i);
                        let bin = ((*pb_ptr.add(row as usize) >> shift) & 0xFF) as usize;
                        
                        let goes_left = if bin == sentinel as usize { missing } else { *cats.get_unchecked(bin) as usize };

                        *left_buffer.get_unchecked_mut(left_count) = row;
                        *right_buffer.get_unchecked_mut(right_count) = row;

                        left_count += goes_left;
                        right_count += 1 - goes_left;
                    }
                }
                for i in mid..n {
                    unsafe {
                        let row = *indices.get_unchecked(i);
                        let bin = ((*pb_ptr.add(row as usize) >> shift) & 0xFF) as usize;
                        let goes_left = if bin == sentinel as usize { missing } else { *cats.get_unchecked(bin) as usize };

                        *left_buffer.get_unchecked_mut(left_count) = row;
                        *right_buffer.get_unchecked_mut(right_count) = row;

                        left_count += goes_left;
                        right_count += 1 - goes_left;
                    }
                }
            }
        }

        indices[..left_count].copy_from_slice(&left_buffer[..left_count]);
        indices[left_count..].copy_from_slice(&right_buffer[..right_count]);

        left_count
    }
}

/// Gather `grad_hess[indices[i]]` into `out[i]` for `i` in `indices`.
#[inline(always)]
fn gather_gradients(indices: &[u32], grad_hess: &[[f32; 2]], out: &mut [[f32; 2]]) {
    const PREFETCH_DIST: usize = 16;
    let n = indices.len();
    let mid = n.saturating_sub(PREFETCH_DIST);
    let gh_ptr = grad_hess.as_ptr();

    for i in 0..mid {
        unsafe {
            let prefetch_index = *indices.get_unchecked(i + PREFETCH_DIST) as usize;
            prefetch(gh_ptr.add(prefetch_index));
            let index = *indices.get_unchecked(i) as usize;
            *out.get_unchecked_mut(i) = *gh_ptr.add(index);
        }
    }
    for i in mid..n {
        unsafe {
            let index = *indices.get_unchecked(i) as usize;
            *out.get_unchecked_mut(i) = *gh_ptr.add(index);
        }
    }
}

#[inline(always)]
pub fn calculate_value(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    if l1 == 0.0 {
        -g / (h + l2)
    } else {
        let d = (g.abs() - l1).max(0.0);
        -g.signum() * d / (h + l2)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn leaf(value: f64) -> Node {
        Node::Leaf { value }
    }

    fn numeric_split(feature: usize, threshold: f64, left_child: usize, right_child: usize, missing_goes_left: bool) -> Node {
        Node::Internal {
            left_child: left_child as u32,
            right_child: right_child as u32,
            split_feature: feature as u32,
            threshold: FinalThreshold::Numeric(threshold),
            missing_goes_left,
        }
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
            Node::Internal {
                left_child: 1,
                right_child: 2,
                split_feature: 0,
                threshold: FinalThreshold::Categorical(vec![true, false, true]),
                missing_goes_left: false,
            },
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
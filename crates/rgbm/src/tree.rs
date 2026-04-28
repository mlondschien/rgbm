// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use arrow::array::{Array, PrimitiveArray};
use arrow::datatypes::Float64Type;
use rayon::prelude::*;

use crate::bin::FeatureBinner;
use crate::dataset::Dataset;
use crate::histogram::{BinSplit, Histograms, SplitInfo};
use crate::parameters::BoosterParameters;
use crate::histogram::calculate_score;
use crate::utils::prefetch;

/// Reusable scratch buffers for tree fitting, owned by the Booster and passed in each
/// iteration to avoid repeated allocation of O(num_rows) memory.
pub struct TreeWorkspace {
    pub leaf_indices: Vec<u32>,
    pub left_buffer: Vec<u32>,
    pub right_buffer: Vec<u32>,
    pub all_indices: Vec<u32>,
    pub ordered_gh: Vec<[f32; 2]>,
    pub partition_flags: Vec<bool>,
}

impl TreeWorkspace {
    pub fn new(num_rows: usize) -> Self {
        Self {
            leaf_indices: vec![0u32; num_rows],
            left_buffer: vec![0u32; num_rows],
            right_buffer: vec![0u32; num_rows],
            all_indices: (0..num_rows as u32).collect(),
            ordered_gh: vec![[0.0f32; 2]; num_rows],
            partition_flags: vec![false; num_rows],
        }
    }
}

#[derive(Clone)]
pub enum Threshold {
    Numeric(f64),
    Categorical([u64; 4]),
}

impl Threshold {
    pub fn from_bin_split(bin_split: &BinSplit, binner: &FeatureBinner) -> Self {
        match bin_split {
            BinSplit::Numeric(bin) => Threshold::Numeric(match binner {
                FeatureBinner::Numerical(upper_bounds) => upper_bounds[*bin as usize],
                FeatureBinner::Categorical(_) => panic!("numeric split on categorical feature"),
            }),
            BinSplit::Categorical(gl) => {
                let mut bitset = [0u64; 4];
                for (i, &goes_left) in gl.iter().enumerate() {
                    if goes_left {
                        bitset[i / 64] |= 1 << (i % 64);
                    }
                }
                Threshold::Categorical(bitset)
            }
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
        threshold: Threshold,
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
    pub leaf_index: usize,
    pub start: usize,
    pub len: usize,
    pub depth: usize,
    pub histograms: Histograms,
    pub best_split: SplitInfo,
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
                &leaf.best_split, &mut workspace.left_buffer, &mut workspace.right_buffer, pool, &mut workspace.partition_flags);

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
                threshold: Threshold::from_bin_split(
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
                        Threshold::Numeric(t) => {
                            let col = numeric_columns[*split_feature as usize].unwrap();
                            if col.is_null(row) {
                                *missing_goes_left
                            } else {
                                let val = col.value(row);
                                if val.is_nan() { *missing_goes_left } else { val <= *t }
                            }
                        }
                        Threshold::Categorical(bitset) => {
                            let col = categorical_columns[*split_feature as usize].as_ref().unwrap();
                            let bin = col[row] as usize;
                            if bin < 256 {
                                (bitset[bin / 64] >> (bin % 64)) & 1 == 1
                            } else {
                                *missing_goes_left
                            }
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

    /// Separate `indices` into left/right based on split. Indices that belong left go
    /// to the front of `indices`, those that belong right go to the back. Returns the
    /// number of indices that go left. Uses buffers instead of in-place swapping to
    /// scrambling the order of rows. That is, the indices are ordered within leafs.
    pub fn partition_indices(
        &self,
        dataset: &Dataset,
        indices: &mut [u32],
        split: &SplitInfo,
        left_buffer: &mut [u32],
        right_buffer: &mut [u32],
        pool: Option<&rayon::ThreadPool>,
        flags: &mut [bool],
    ) -> usize {
        let n = indices.len();
        let bundle = &dataset.feature_bundles[split.feature_index / 4];
        let shift = (split.feature_index % 4) * 8;
        let bins = &bundle.packed_bins;
        let sentinel = (dataset.feature_binners[split.feature_index].num_bins() - 1) as u8;
        let missing = split.missing_goes_left;

        let check_left = |row: u32| -> bool {
            let bin = ((bins[row as usize] >> shift) & 0xFF) as u8;
            if bin == sentinel { missing } else {
                match &split.threshold {
                    BinSplit::Numeric(t) => bin <= *t as u8,
                    BinSplit::Categorical(cats) => cats[bin as usize],
                }
            }
        };

        // Algorithms differ between parallel and sequential execution. For parallel,
        // 3 steps: (i) Evaluate check_left in parallel and store results in `flags`.
        // Get `counts` containing the number of left rows per chunk. (ii) Prepare
        // output slices for each chunk based on `counts`. (iii) Fill output buffers in
        // parallel using `flags`
        let total_left = if let Some(pool) = pool.filter(|_| n > 2048) {
            let n_threads = pool.current_num_threads();
            let chunk_size = (n / (n_threads * 32)).max(1024);

            // (i) Write `check_left` into `flags` and get counts of left rows per chunk
            let counts: Vec<usize> = pool.install(|| {
                indices.par_chunks(chunk_size)
                    .zip(flags[..n].par_chunks_mut(chunk_size))
                    .map(|(chunk, chunk_flags)| {
                        let mut count = 0;
                        for (j, &row) in chunk.iter().enumerate() {
                            let left = check_left(row);
                            chunk_flags[j] = left;
                            if left { count += 1; }
                        }
                        count
                    }).collect()
            });

            // (ii) Sequentially, prepare output slices based on `counts`. Use
            // `split_at_mut` to write into (disjoint) slices of left/right slices
            // without unsafe indexing.
            let mut left_slices = Vec::new();
            let mut right_slices = Vec::new();
            let mut left_remaining = &mut *left_buffer;
            let mut right_remaining = &mut *right_buffer;
            for (i, &left_count) in counts.iter().enumerate() {
                // this is chunk_size except for the last chunk, which may be smaller.
                let chunk_length: usize = (i * chunk_size + chunk_size).min(n) - i * chunk_size;
                let (left_current, left_tail) = left_remaining.split_at_mut(left_count);
                let (right_current, right_tail) = right_remaining.split_at_mut(chunk_length - left_count);
                left_slices.push(left_current);
                right_slices.push(right_current);
                left_remaining = left_tail;
                right_remaining = right_tail;
            }

            // (iii) In parallel, fill the output buffers based on `flags`. Each chunk
            // writes to its output slice.
            pool.install(|| {
                indices.par_chunks(chunk_size)
                    .zip(left_slices)
                    .zip(right_slices)
                    .zip(flags[..n].par_chunks(chunk_size))
                    .for_each(|(((indices_chunk, left_out), right_out), flags_chunk)| {
                        let (mut left_idx, mut right_idx) = (0, 0);
                        for (j, &row) in indices_chunk.iter().enumerate() {
                            if flags_chunk[j] {
                                left_out[left_idx] = row;
                                left_idx += 1;
                            } else {
                                right_out[right_idx] = row;
                                right_idx += 1;
                            }
                        }
                    });
            });
            counts.iter().sum()
        } else {
            // much simpler sequential partitioning.
            let (mut l, mut r) = (0, 0);
            for &row in indices.iter() {
                if check_left(row) {
                    left_buffer[l] = row;
                    l += 1;
                } else {
                    right_buffer[r] = row;
                    r += 1;
                }
            }
            l
        };

        indices[..total_left].copy_from_slice(&left_buffer[..total_left]);
        indices[total_left..].copy_from_slice(&right_buffer[..n - total_left]);
        total_left
    }
}

/// Gather `grad_hess[indices[i]]` into `out[i]` for `i` in `indices`.
#[inline(always)]
fn gather_gradients(indices: &[u32], grad_hess: &[[f32; 2]], out: &mut [[f32; 2]]) {
    const PREFETCH_DIST: usize = 64;
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
            threshold: Threshold::Numeric(threshold),
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
        let mut bitset = [0u64; 4];
        bitset[0] = (1 << 0) | (1 << 2);

        let tree = Tree { nodes: vec![
            Node::Internal {
                left_child: 1,
                right_child: 2,
                split_feature: 0,
                threshold: Threshold::Categorical(bitset),
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

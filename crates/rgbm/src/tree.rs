// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use rayon::prelude::*;

use crate::dataset::Dataset;
use crate::histogram::{Histograms, SplitInfo, Threshold};
use crate::parameters::BoosterParameters;

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
        Self {
            nodes: Vec::with_capacity(2 * max_leaves),
        }
    }

    /// Fit the tree, returning each row's leaf node index.
    pub fn fit(
        &mut self,
        dataset: &Dataset,
        grad_hess: &[[f32; 2]],
        p: &BoosterParameters,
        pool: Option<&rayon::ThreadPool>,
    ) -> Vec<u32> {
        self.nodes.clear();

        let num_rows = dataset.num_rows;
        let mut all_indices: Vec<u32> = (0..num_rows as u32).collect();

        let mut left_buffer = vec![0u32; num_rows];
        let mut right_buffer = vec![0u32; num_rows];
        let mut partition_flags = vec![false; num_rows];
        let mut ordered_gh = vec![[0.0f32; 2]; num_rows];

        let mut active_leaves: Vec<ActiveLeaf> = Vec::new();

        // Leaves that won't be split further, stored as (start, len, node_idx).
        let mut finished_leaves: Vec<(usize, usize, usize)> = Vec::new();

        let root_histograms =
            Histograms::build(&dataset.feature_bundles, grad_hess, &all_indices, pool);

        let (root_gradient, root_hessian) = grad_hess.iter().fold((0.0, 0.0), |(g, h), gh| {
            (g + gh[0] as f64, h + gh[1] as f64)
        });

        self.push_leaf(
            &mut active_leaves,
            &mut finished_leaves,
            Some(root_histograms),
            root_gradient,
            root_hessian,
            0,
            dataset.num_rows,
            0,
            p,
            pool,
        );
        let mut num_leaves = 1;

        while num_leaves < p.max_leaves && !active_leaves.is_empty() {
            // Leaf-wise: highest-gain leaf first..
            // Depth-wise: shallowest leaf first, ties broken by highest gain
            let (idx, _) = if p.leaf_wise {
                active_leaves
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.best_split.gain.total_cmp(&b.best_split.gain))
                    .unwrap()
            } else {
                active_leaves
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.depth
                            .cmp(&b.depth)
                            .then_with(|| b.best_split.gain.total_cmp(&a.best_split.gain))
                    })
                    .unwrap()
            };
            let leaf = active_leaves.swap_remove(idx);

            let split_position = self.partition_indices(
                dataset,
                &mut all_indices[leaf.start..leaf.start + leaf.len],
                &leaf.best_split,
                &mut left_buffer,
                &mut right_buffer,
                pool,
                &mut partition_flags,
            );

            let left_start = leaf.start;
            let left_len = split_position;
            let right_start = leaf.start + split_position;
            let right_len = leaf.len - split_position;

            // Skip building histograms if neither child can be split further.
            let min_hessian = p.min_sum_hessian_in_leaf * 2.0;
            let both_terminal = leaf.depth + 1 >= p.max_depth
                || num_leaves + 1 >= p.max_leaves
                || (leaf.best_split.left_hessian < min_hessian
                    && leaf.best_split.right_hessian < min_hessian);

            // Build the smaller child directly, derive the larger by subtracting
            let (left_histograms, right_histograms) = if both_terminal {
                (None, None)
            } else if left_len < right_len {
                let left_indices = &all_indices[left_start..left_start + left_len];
                let ordered_grad_hess = &mut ordered_gh[..left_len];
                gather_grad_hess(grad_hess, left_indices, ordered_grad_hess, pool);
                let left_histograms = Histograms::build(
                    &dataset.feature_bundles,
                    ordered_grad_hess,
                    left_indices,
                    pool,
                );
                let mut right_histograms = leaf.histograms;
                right_histograms.subtract(&left_histograms);
                (Some(left_histograms), Some(right_histograms))
            } else {
                let right_indices = &all_indices[right_start..right_start + right_len];
                let ordered_grad_hess = &mut ordered_gh[..right_len];
                gather_grad_hess(grad_hess, right_indices, ordered_grad_hess, pool);
                let right_histograms = Histograms::build(
                    &dataset.feature_bundles,
                    ordered_grad_hess,
                    right_indices,
                    pool,
                );
                let mut left_histograms = leaf.histograms;
                left_histograms.subtract(&right_histograms);
                (Some(left_histograms), Some(right_histograms))
            };

            let left_node_idx = self.push_leaf(
                &mut active_leaves,
                &mut finished_leaves,
                left_histograms,
                leaf.best_split.left_gradient,
                leaf.best_split.left_hessian,
                left_start,
                left_len,
                leaf.depth + 1,
                p,
                pool,
            );
            let right_node_idx = self.push_leaf(
                &mut active_leaves,
                &mut finished_leaves,
                right_histograms,
                leaf.best_split.right_gradient,
                leaf.best_split.right_hessian,
                right_start,
                right_len,
                leaf.depth + 1,
                p,
                pool,
            );

            self.nodes[leaf.leaf_index] = Node::Internal {
                left_child: left_node_idx as u32,
                right_child: right_node_idx as u32,
                split_feature: leaf.best_split.feature_index as u32,
                threshold: leaf.best_split.threshold,
            };
            num_leaves += 1;
        }

        let mut leaf_indices = vec![0u32; num_rows];
        let active = active_leaves.iter().map(|l| (l.start, l.len, l.leaf_index));
        for (start, len, node_idx) in finished_leaves.into_iter().chain(active) {
            for &row in &all_indices[start..start + len] {
                leaf_indices[row as usize] = node_idx as u32;
            }
        }

        leaf_indices
    }

    /// Zero-allocation, lock-free evaluation of a single row. For speed.
    ///
    /// `columns[f]` is the per-row binned values for feature `f`; `sentinels[f]` is
    /// the bin index used to encode missing/null/NaN for that feature.
    #[inline(always)]
    pub fn predict_row(&self, row: usize, columns: &[&[u8]], sentinels: &[u8]) -> f64 {
        let mut idx = 0;
        loop {
            match unsafe { self.nodes.get_unchecked(idx) } {
                Node::Leaf { value } => return *value,
                Node::Internal {
                    left_child,
                    right_child,
                    split_feature,
                    threshold,
                } => {
                    let bin = columns[*split_feature as usize][row];
                    let goes_left = match threshold {
                        Threshold::Numeric {
                            bin: t,
                            missing_goes_left,
                        } => {
                            if bin == sentinels[*split_feature as usize] {
                                *missing_goes_left
                            } else {
                                bin <= *t
                            }
                        }
                        Threshold::Categorical(cats) => cats[bin as usize],
                    };
                    idx = if goes_left {
                        *left_child as usize
                    } else {
                        *right_child as usize
                    };
                }
            }
        }
    }

    fn push_leaf(
        &mut self,
        active_leaves: &mut Vec<ActiveLeaf>,
        finished_leaves: &mut Vec<(usize, usize, usize)>,
        histograms: Option<Histograms>,
        gradient: f64,
        hessian: f64,
        start: usize,
        len: usize,
        depth: usize,
        p: &BoosterParameters,
        pool: Option<&rayon::ThreadPool>,
    ) -> usize {
        let value = calculate_value(gradient, hessian, p.lambda_l1, p.lambda_l2) * p.learning_rate;
        let node_idx = self.nodes.len();

        self.nodes.push(Node::Leaf { value });

        let best_split = match histograms {
            Some(histograms)
                if depth < p.max_depth && hessian >= p.min_sum_hessian_in_leaf * 2.0 =>
            {
                histograms
                    .find_best_split(p, pool)
                    .map(|best_split| (histograms, best_split))
            }
            _ => None,
        };

        match best_split {
            Some((histograms, best_split)) => active_leaves.push(ActiveLeaf {
                leaf_index: node_idx,
                start,
                len,
                depth,
                histograms,
                best_split,
            }),
            None => finished_leaves.push((start, len, node_idx)),
        }
        node_idx
    }

    /// Separate `indices` into left/right based on split. Indices that belong left go
    /// to the front of `indices`, those that belong right go to the back. Returns the
    /// number of indices that go left. Uses buffers instead of in-place swapping to
    /// avoid scrambling the order of rows. That is, the indices are ordered within leafs.
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
        let bundle = &dataset.feature_bundles[split.feature_index / 8];
        let shift = (split.feature_index % 8) * 8;
        let bins = &bundle.packed_bins;
        let sentinel = (dataset.feature_binners[split.feature_index].num_bins() - 1) as u8;

        let check_left = |row: u32| -> bool {
            let bin = ((bins[row as usize] >> shift) & 0xFF) as u8;
            match &split.threshold {
                Threshold::Numeric {
                    bin: t,
                    missing_goes_left,
                } => {
                    if bin == sentinel {
                        *missing_goes_left
                    } else {
                        bin <= *t
                    }
                }
                Threshold::Categorical(cats) => cats[bin as usize],
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
                indices
                    .par_chunks(chunk_size)
                    .zip(flags[..n].par_chunks_mut(chunk_size))
                    .map(|(chunk, chunk_flags)| {
                        let mut count = 0;
                        for (j, &row) in chunk.iter().enumerate() {
                            let left = check_left(row);
                            chunk_flags[j] = left;
                            if left {
                                count += 1;
                            }
                        }
                        count
                    })
                    .collect()
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
                let (right_current, right_tail) =
                    right_remaining.split_at_mut(chunk_length - left_count);
                left_slices.push(left_current);
                right_slices.push(right_current);
                left_remaining = left_tail;
                right_remaining = right_tail;
            }

            // (iii) In parallel, fill the output buffers based on `flags`. Each chunk
            // writes to its output slice.
            pool.install(|| {
                indices
                    .par_chunks(chunk_size)
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

/// Gather `grad_hess[row]` for each row in `indices` into `out`.
fn gather_grad_hess(
    grad_hess: &[[f32; 2]],
    indices: &[u32],
    out: &mut [[f32; 2]],
    pool: Option<&rayon::ThreadPool>,
) {
    match pool.filter(|_| indices.len() > 2048) {
        Some(pool) => pool.install(|| {
            out.par_iter_mut()
                .zip(indices.par_iter())
                .with_min_len(1024)
                .for_each(|(out, &row)| *out = grad_hess[row as usize]);
        }),
        None => {
            for (out, &row) in out.iter_mut().zip(indices) {
                *out = grad_hess[row as usize];
            }
        }
    }
}

#[inline(always)]
fn calculate_value(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
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

    fn leaf(value: f64) -> Node {
        Node::Leaf { value }
    }

    fn numeric_split(
        feature: usize,
        bin: u8,
        left_child: usize,
        right_child: usize,
        missing_goes_left: bool,
    ) -> Node {
        Node::Internal {
            left_child: left_child as u32,
            right_child: right_child as u32,
            split_feature: feature as u32,
            threshold: Threshold::Numeric {
                bin,
                missing_goes_left,
            },
        }
    }

    #[test]
    fn test_predict_numeric_stump() {
        // bin <= 5 → -1.0, else → 1.0. Sentinel = 9; bit unset → missing → right.
        let tree = Tree {
            nodes: vec![numeric_split(0, 5, 1, 2, false), leaf(-1.0), leaf(1.0)],
        };
        let col0: &[u8] = &[3, 7];
        let cols = [col0];
        let sentinels = [9u8];
        assert_eq!(tree.predict_row(0, &cols, &sentinels), -1.0); // bin 3 → left
        assert_eq!(tree.predict_row(1, &cols, &sentinels), 1.0); // bin 7 → right
    }

    #[test]
    fn test_predict_missing_goes_left() {
        let tree = Tree {
            nodes: vec![
                numeric_split(0, 5, 1, 2, true), // missing → left
                leaf(-1.0),
                leaf(1.0),
            ],
        };
        let col0: &[u8] = &[9]; // bin 9 = sentinel
        let cols = [col0];
        let sentinels = [9u8];
        assert_eq!(tree.predict_row(0, &cols, &sentinels), -1.0);
    }

    #[test]
    fn test_predict_categorical() {
        // categories 0 and 2 go left, category 1 goes right
        let tree = Tree {
            nodes: vec![
                Node::Internal {
                    left_child: 1,
                    right_child: 2,
                    split_feature: 0,
                    threshold: Threshold::Categorical(vec![true, false, true]),
                },
                leaf(-1.0),
                leaf(1.0),
            ],
        };
        let col0: &[u8] = &[0, 1, 2];
        let cols = [col0];
        let sentinels = [3u8]; // unused for categorical
        assert_eq!(tree.predict_row(0, &cols, &sentinels), -1.0); // cat 0 → left
        assert_eq!(tree.predict_row(1, &cols, &sentinels), 1.0); // cat 1 → right
        assert_eq!(tree.predict_row(2, &cols, &sentinels), -1.0); // cat 2 → left
    }

    #[test]
    fn test_partition_indices_numeric() {
        use crate::dataset::Dataset;
        use crate::histogram::SplitInfo;
        use crate::parameters::DatasetParameters;
        use arrow::array::{Float64Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        // 10 rows, 1 numeric feature. Build dataset, then partition by bin <= some threshold.
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(values))]).unwrap();
        let labels = Float64Array::from(vec![0.0; 10]);
        let dataset = Dataset::from_arrow(
            &batch,
            &labels,
            None,
            None,
            &DatasetParameters {
                min_data_in_bin: 1,
                ..DatasetParameters::default()
            },
        );

        // Build a bin <= 4 split. Bins 0..=4 go left, others right.
        let split = SplitInfo {
            gain: 1.0,
            threshold: Threshold::Numeric {
                bin: 4,
                missing_goes_left: false,
            },
            feature_index: 0,
            left_gradient: 0.0,
            left_hessian: 0.0,
            right_gradient: 0.0,
            right_hessian: 0.0,
        };

        let tree = Tree::new(1);
        let mut indices: Vec<u32> = (0..10).collect();
        let mut left_buf = vec![0u32; 10];
        let mut right_buf = vec![0u32; 10];
        let mut flags = vec![false; 10];

        let n_left = tree.partition_indices(
            &dataset,
            &mut indices,
            &split,
            &mut left_buf,
            &mut right_buf,
            None,
            &mut flags,
        );

        // Left rows must have feature-bin <= 4; right rows must have bin > 4.
        let bundle = &dataset.feature_bundles[0];
        for &row in &indices[..n_left] {
            assert!(
                (bundle.packed_bins[row as usize] & 0xFF) as u8 <= 4,
                "left row has bin > 4"
            );
        }
        for &row in &indices[n_left..] {
            assert!(
                (bundle.packed_bins[row as usize] & 0xFF) as u8 > 4,
                "right row has bin <= 4"
            );
        }
        assert_eq!(n_left + (10 - n_left), 10);
    }

    #[test]
    fn test_predict_deep_tree() {
        // f0 splits at bin 5; if left, f1 splits at bin 5
        // [f0=3, f1=3]→-2, [f0=3, f1=7]→-1, [f0=7,*]→1
        let tree = Tree {
            nodes: vec![
                numeric_split(0, 5, 1, 4, false),
                numeric_split(1, 5, 2, 3, false),
                leaf(-2.0),
                leaf(-1.0),
                leaf(1.0),
            ],
        };
        let c0: &[u8] = &[3, 3, 7];
        let c1: &[u8] = &[3, 7, 3];
        let cols = [c0, c1];
        let sentinels = [9u8, 9u8];
        assert_eq!(tree.predict_row(0, &cols, &sentinels), -2.0);
        assert_eq!(tree.predict_row(1, &cols, &sentinels), -1.0);
        assert_eq!(tree.predict_row(2, &cols, &sentinels), 1.0);
    }
}

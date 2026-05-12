// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

//! Histogram-based gradient and hessian accumulation, and best-split finding.

use rayon::prelude::*;
use crate::dataset::FeatureBundle;
use crate::parameters::BoosterParameters;
use crate::utils::prefetch;

#[derive(Clone, Default, Debug)]
#[repr(C)]  // group fields into 16 bytes for optimal SIMD processing
pub struct HistogramBin {
    pub sum_gradients: f64,
    pub sum_hessians: f64,
}

impl HistogramBin {
    #[inline(always)]
    pub fn add(&mut self, gh: [f32; 2]) {
        self.sum_gradients += gh[0] as f64;
        self.sum_hessians += gh[1] as f64;
    }
}

/// `Numeric { bin, missing_goes_left }`: rows with bin index `<= bin` go left.
///  The sentinel/missing bin is routed by `missing_goes_left`.
/// `Categorical(goes_left)`: one bool per bin. `goes_left[bin]` says which side
///  the row goes. The sentinel bin's slot encodes missing routing.
#[derive(Clone, Debug)]
pub enum Threshold {
    Numeric { bin: u8, missing_goes_left: bool },
    Categorical(Vec<bool>),
}

pub struct SplitInfo {
    pub gain: f64,
    pub threshold: Threshold,
    pub feature_index: usize,
}

pub struct Histograms {
    /// The bins for all features in a contiguous block of memory.
    pub bins: Vec<HistogramBin>,
    /// The starting index of each feature's bins within the `bins` array.
    pub offsets: Vec<usize>,
    pub is_categorical: Vec<bool>,
}

impl Histograms {
    pub fn zeros(bundles: &[FeatureBundle]) -> Self {
        let num_features = bundles.iter().map(|b| b.count).sum();
        let mut offsets = Vec::with_capacity(num_features + 1);
        let mut is_categorical = Vec::with_capacity(num_features);
        
        let mut offset = 0;
        for bundle in bundles {
            for i in 0..bundle.count {
                offsets.push(offset);
                offset += bundle.num_bins[i];
                is_categorical.push(bundle.is_categorical[i]);
            }
        }
        offsets.push(offset); // allow indexing offset[idx]...offset[idx+1].

        Self {
            bins: vec![HistogramBin::default(); offset],
            offsets,
            is_categorical,
        }
    }


    /// Build histograms for all features in a bundle in one pass over row_indices.
    /// One 64-bit load per row replaces up to 8 separate byte loads.
    /// SAFETY: bin values are in 0..num_bins[i] (sentinel is at index num_bins[i] - 1),
    /// histograms are sized num_bins[i].
    pub fn build_into(bundle: &FeatureBundle, packed_gh: &[[f32; 2]], row_indices: &[u32], bins: &mut [HistogramBin]) {
        let p0 = bins.as_mut_ptr();

        const PREFETCH_DIST: usize = 32;
        // mid is even so the 2-way unrolled main loop has no remainder.
        let mid = (row_indices.len().saturating_sub(PREFETCH_DIST) / 2) * 2;

        // Shared loop body for all bundle sizes. Pointers `p1..p7` and shift values
        // are supplied per arm; `p0` is always the start of `bins`.
        macro_rules! arm {
            ($($pi:ident $shift:literal),* $(,)?) => {
                let mut i = 0;
                while i < mid {
                    unsafe {
                        prefetch(bundle.packed_bins.get_unchecked(*row_indices.get_unchecked(i + PREFETCH_DIST) as usize));
                        prefetch(bundle.packed_bins.get_unchecked(*row_indices.get_unchecked(i + PREFETCH_DIST + 1) as usize));
                        let row0 = *row_indices.get_unchecked(i) as usize;
                        let row1 = *row_indices.get_unchecked(i + 1) as usize;
                        let gh0 = *packed_gh.get_unchecked(i);
                        let gh1 = *packed_gh.get_unchecked(i + 1);
                        let pk0 = *bundle.packed_bins.get_unchecked(row0);
                        let pk1 = *bundle.packed_bins.get_unchecked(row1);
                        
                        
                        (*p0.add((pk0 & 0xFF) as usize)).add(gh0);
                        // (*p1.add(((pk0 >> 8) & 0xFF) as usize)).add(gh0);                                                                                                                                      
                        // ...
                        // (*p7.add(((pk0 >> 56) & 0xFF) as usize)).add(gh0);
                        $((*$pi.add(((pk0 >> $shift) & 0xFF) as usize)).add(gh0);)*

                        (*p0.add((pk1 & 0xFF) as usize)).add(gh1);
                        // (*p1.add(((pk1 >> 8) & 0xFF) as usize)).add(gh1);
                        // ...                                                                                                                                   
                        // (*p7.add(((pk1 >> 56) & 0xFF) as usize)).add(gh1); 
                        $((*$pi.add(((pk1 >> $shift) & 0xFF) as usize)).add(gh1);)*
                    }
                    i += 2;
                }
                for i in mid..row_indices.len() {
                    unsafe {
                        let row = *row_indices.get_unchecked(i) as usize;
                        let gh = *packed_gh.get_unchecked(i);
                        let pk = *bundle.packed_bins.get_unchecked(row);
                        (*p0.add((pk & 0xFF) as usize)).add(gh);
                        $((*$pi.add(((pk >> $shift) & 0xFF) as usize)).add(gh);)*
                    }
                }
            };
        }

        match bundle.count {
            8 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                let p3 = unsafe { p2.add(bundle.num_bins[2]) };
                let p4 = unsafe { p3.add(bundle.num_bins[3]) };
                let p5 = unsafe { p4.add(bundle.num_bins[4]) };
                let p6 = unsafe { p5.add(bundle.num_bins[5]) };
                let p7 = unsafe { p6.add(bundle.num_bins[6]) };
                arm!(p1 8, p2 16, p3 24, p4 32, p5 40, p6 48, p7 56);
            }
            7 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                let p3 = unsafe { p2.add(bundle.num_bins[2]) };
                let p4 = unsafe { p3.add(bundle.num_bins[3]) };
                let p5 = unsafe { p4.add(bundle.num_bins[4]) };
                let p6 = unsafe { p5.add(bundle.num_bins[5]) };
                arm!(p1 8, p2 16, p3 24, p4 32, p5 40, p6 48);
            }
            6 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                let p3 = unsafe { p2.add(bundle.num_bins[2]) };
                let p4 = unsafe { p3.add(bundle.num_bins[3]) };
                let p5 = unsafe { p4.add(bundle.num_bins[4]) };
                arm!(p1 8, p2 16, p3 24, p4 32, p5 40);
            }
            5 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                let p3 = unsafe { p2.add(bundle.num_bins[2]) };
                let p4 = unsafe { p3.add(bundle.num_bins[3]) };
                arm!(p1 8, p2 16, p3 24, p4 32);
            }
            4 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                let p3 = unsafe { p2.add(bundle.num_bins[2]) };
                arm!(p1 8, p2 16, p3 24);
            }
            3 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                let p2 = unsafe { p1.add(bundle.num_bins[1]) };
                arm!(p1 8, p2 16);
            }
            2 => {
                let p1 = unsafe { p0.add(bundle.num_bins[0]) };
                arm!(p1 8);
            }
            1 => {
                arm!();
            }
            _ => unreachable!("Bundles must have between 1 and 8 features"),
        }
    }


    pub fn build(
        bundles: &[FeatureBundle],
        grad_hess: &[[f32; 2]],
        indices: &[u32],
        pool: Option<&rayon::ThreadPool>
    ) -> Self {
        let mut hists = Self::zeros(bundles);

        // Slice hists.bins into mutable slices for each bundle, to be handed to each
        // thread. This needs some extra code to make the borrow checker happy using
        // safe rust.
        let mut remaining_bins = hists.bins.as_mut_slice();
        let mut bin_slices = Vec::with_capacity(bundles.len());

        for bundle in bundles {
            let len = bundle.num_bins.iter().take(bundle.count).sum::<usize>();
            let (curr, rest) = remaining_bins.split_at_mut(len);
            bin_slices.push(curr);
            remaining_bins = rest;
        }

        match pool {
            None => {
                for (bundle, local_bins) in bundles.iter().zip(bin_slices) {
                    Self::build_into(bundle, grad_hess, indices, local_bins);
                }
            }
            Some(pool) => pool.install(|| {
                bundles.par_iter()
                    .zip(bin_slices)
                    .for_each(|(bundle, local_bins)| {
                        Self::build_into(bundle, grad_hess, indices, local_bins);
                    });
            }),
        }

        hists
    }


    pub fn subtract(&mut self, other: &Self) {
        debug_assert_eq!(self.bins.len(), other.bins.len());

        for (s, o) in self.bins.iter_mut().zip(other.bins.iter()) {
            s.sum_gradients -= o.sum_gradients;
            s.sum_hessians -= o.sum_hessians;
        }
    }

    /// Find the best numeric split by scanning bins left to right.
    pub fn find_best_numeric_split(
        bins: &[HistogramBin],
        total_gradient: f64,
        total_hessian: f64,
        parent_score: f64,
        parameters: &BoosterParameters,
    ) -> Option<SplitInfo> {
        let sentinel_bin = &bins.last().unwrap();

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;

        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;
        let mut best_missing_goes_left = false;

        // Code duplication: Two loops with the same gain calculation but with the
        // branch on the outside.
        if sentinel_bin.sum_hessians == 0.0 {
            // If there's two bins: One sentinel and one regular and no values in the
            // sentinel bin, there's nothing to split.
            if bins.len() < 3 { return None; }
            // Safe due to the check above.
            for t in 0..bins.len() - 2 {
                let bin = &bins[t];
                left_gradient += bin.sum_gradients;
                left_hessian += bin.sum_hessians;

                let right_gradient = total_gradient - left_gradient;
                let right_hessian = total_hessian - left_hessian;
                // Given a feature with no missing values, we still need to decide how
                // to route missing values at predict. Heuristic: Route them to the
                // "bigger" side.
                let missing_goes_left = left_hessian > right_hessian;
                
                evaluate_split(
                    left_gradient, left_hessian, right_gradient, right_hessian,
                    missing_goes_left, t, parameters, 
                    &mut best_score, &mut best_threshold, &mut best_missing_goes_left
                );
            }
        } else {
            if bins.len() < 2 { return None; }
            // One additional split to check compared to above: All missing values go
            // left, everything else goes right.
            for t in 0..bins.len() - 1 {
                // first compute score for missing_goes_left = false.
                left_gradient += bins[t].sum_gradients;
                left_hessian += bins[t].sum_hessians;

                let right_gradient = total_gradient - left_gradient;
                let right_hessian = total_hessian - left_hessian;

                evaluate_split(
                    left_gradient, left_hessian, right_gradient, right_hessian,
                    false, t, parameters, 
                    &mut best_score, &mut best_threshold, &mut best_missing_goes_left
                );

                // now compute score for missing_goes_left = true.
                let left_gradient_plus_sentinel = left_gradient + sentinel_bin.sum_gradients;
                let left_hessian_plus_sentinel = left_hessian + sentinel_bin.sum_hessians;
                let right_gradient_minus_sentinel = right_gradient - sentinel_bin.sum_gradients;
                let right_hessian_minus_sentinel = right_hessian - sentinel_bin.sum_hessians;

                evaluate_split(
                    left_gradient_plus_sentinel, left_hessian_plus_sentinel,
                    right_gradient_minus_sentinel, right_hessian_minus_sentinel,
                    true, t, parameters, 
                    &mut best_score, &mut best_threshold, &mut best_missing_goes_left
                );
            }
        }

        let gain = best_score - parent_score;
        if gain <= parameters.min_gain_to_split { return None; }

        Some(SplitInfo {
            gain,
            threshold: Threshold::Numeric {
                bin: best_threshold as u8,
                missing_goes_left: best_missing_goes_left,
            },
            feature_index: 0, // to be filled in by caller
        })
    }

    /// Find the best categorical split by sorting bins by gradient ratio.
    /// 
    /// Instead of checking all 2^num_bins subsets, we first sort categories by their
    /// gradient/hessian ratio. Then we only need to check splits between sorted
    /// categories. This is exact according to Fisher, W. D. (1958).
    pub fn find_best_categorical_split(
        bins: &[HistogramBin],
        total_gradient: f64,
        total_hessian: f64,
        parent_score: f64,
        parameters: &BoosterParameters,
    ) -> Option<SplitInfo> {
        let num_bins = bins.len();

        // We treat missing values as just another category.
        let mut categorical_order: Vec<(f64, usize)> = Vec::with_capacity(num_bins);
        for k in 0..num_bins {
            let bin = &bins[k];
            if bin.sum_hessians > 0.0 {
                let ratio = bin.sum_gradients / (bin.sum_hessians + parameters.lambda_l2);
                categorical_order.push((ratio, k));
            }
        }

        if categorical_order.len() < 2 { return None; }

        // Sort categories to find the optimal contiguous binary partition
        categorical_order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;
        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;
        let mut best_majority_goes_left = false;
    
        for t in 0..categorical_order.len() - 1 {
            let bin = &bins[categorical_order[t].1];
            left_gradient += bin.sum_gradients;
            left_hessian += bin.sum_hessians;

            let right_gradient = total_gradient - left_gradient;
            let right_hessian = total_hessian - left_hessian;

            let majority_goes_left = left_hessian > right_hessian;

            evaluate_split(
                left_gradient, left_hessian, right_gradient, right_hessian,
                majority_goes_left, t, parameters, 
                &mut best_score, &mut best_threshold, &mut best_majority_goes_left
            );
        }

        let gain = best_score - parent_score;
        if gain <= parameters.min_gain_to_split { return None; }

        let mut goes_left = vec![best_majority_goes_left; num_bins];
        for (i, &(_, k)) in categorical_order.iter().enumerate() {
            goes_left[k] = i <= best_threshold;
        }
        // The sentinel/missing bin sits at num_bins - 1; its slot in `goes_left`
        // is the source of truth for missing-value routing.

        Some(SplitInfo {
            gain,
            threshold: Threshold::Categorical(goes_left),
            feature_index: 0, // to be filled in by caller
        })
    }

    pub fn find_best_split(&self, total_gradient: f64, total_hessian: f64, parent_score: f64, p: &BoosterParameters, pool: Option<&rayon::ThreadPool>) -> Option<SplitInfo> {
        let num_features = self.is_categorical.len();
        let map_function = |f: usize| {
            let bins = &self.bins[self.offsets[f]..self.offsets[f + 1]];
            let split_opt = if self.is_categorical[f] {
                Self::find_best_categorical_split(bins, total_gradient, total_hessian, parent_score, p)
            } else {
                Self::find_best_numeric_split(bins, total_gradient, total_hessian, parent_score, p)
            };
            split_opt.map(|mut s| { s.feature_index = f; s })
        };
        let reduce_function = |a: SplitInfo, b: SplitInfo| if a.gain >= b.gain { a } else { b };
        match pool {
            Some(pool) => pool.install(|| {
                (0..num_features).into_par_iter().filter_map(map_function).reduce_with(reduce_function)
            }),
            None => (0..num_features).filter_map(map_function).reduce(reduce_function),
        }
    }
}



/// Score of a leaf node used for gain calculation.
/// ///
/// Branchless implementation for optimal SIMD performance. See also LGBM implementation
#[inline(always)]
pub fn calculate_score(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    let d = (g.abs() - l1).max(0.0);
    d * d / (h + l2)
}



#[inline(always)]
fn evaluate_split(
    left_gradient: f64,
    left_hessian: f64,
    right_gradient: f64,
    right_hessian: f64,
    missing_goes_left: bool,
    threshold_idx: usize,
    parameters: &BoosterParameters,
    best_score: &mut f64,
    best_threshold: &mut usize,
    best_missing_goes_left: &mut bool,
) {
    let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) 
              + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);

    let leaf_constraint = (left_hessian >= parameters.min_sum_hessian_in_leaf) 
                        & (right_hessian >= parameters.min_sum_hessian_in_leaf);
    
    let score = if leaf_constraint { score } else { f64::NEG_INFINITY };

    if score > *best_score {
        *best_score = score;
        *best_threshold = threshold_idx;
        *best_missing_goes_left = missing_goes_left;
    }
}
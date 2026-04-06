//! Histogram-based gradient and hessian accumulation, and best-split finding.

use crate::dataset::FeatureBundle;
use crate::parameters::Parameters;


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


#[derive(Clone, Debug)]
pub enum Threshold {
    Numeric(u32),
    Categorical(Vec<bool>),
}

pub struct SplitInfo {
    pub gain: f64,
    pub missing_goes_left: bool,
    pub threshold: Threshold,
    pub feature_index: usize,
}

/// Histogram for one feature over a set of rows.
pub struct Histogram {
    pub bins: Vec<HistogramBin>,
    pub is_categorical: bool,
}

impl Histogram {
    pub fn zeros(n: usize, is_categorical: bool) -> Self {
        Self { bins: vec![HistogramBin::default(); n], is_categorical }
    }

    pub fn find_best_split(&self, total_gradient: f64, total_hessian: f64, parent_score: f64, p: &Parameters) -> Option<SplitInfo> {
        if self.is_categorical {
            self.find_best_categorical_split(total_gradient, total_hessian, parent_score, p)
        } else {
            self.find_best_numeric_split(total_gradient, total_hessian, parent_score, p)
        }
    }

    /// Build histograms for all features in a bundle in one pass over row_indices.
    /// One 32-bit load per row replaces up to 4 separate byte loads.
    /// SAFETY: bin values are in 0..num_bins[i] (sentinel is at index num_bins[i] - 1),
    /// histograms are sized num_bins[i].
    pub fn build(bundle: &FeatureBundle, grad_hess: &[[f32; 2]], row_indices: &[u32]) -> Vec<Self> {
        // unwrap_or(1) for the case where the bundle has fewer than 4 features.
        let n0 = bundle.num_bins.get(0).copied().unwrap_or(1);
        let n1 = bundle.num_bins.get(1).copied().unwrap_or(1);
        let n2 = bundle.num_bins.get(2).copied().unwrap_or(1);
        let n3 = bundle.num_bins.get(3).copied().unwrap_or(1);

        let mut h0 = vec![HistogramBin::default(); n0];
        let mut h1 = vec![HistogramBin::default(); n1];
        let mut h2 = vec![HistogramBin::default(); n2];
        let mut h3 = vec![HistogramBin::default(); n3];

        let p0 = h0.as_mut_ptr();
        let p1 = h1.as_mut_ptr();
        let p2 = h2.as_mut_ptr();
        let p3 = h3.as_mut_ptr();

        // Use chunks_exact to guarantee chunk size and avoid bounds checks. We handle
        // the possible leftover chunk below.
        // with 2x unrolling and 4 features per chunk, data/pointers in one loop fit
        // comfortably into modern CPU registers.
        let mut chunks = row_indices.chunks_exact(2);

        for chunk in &mut chunks {
            let row0 = chunk[0] as usize;
            let row1 = chunk[1] as usize;
            unsafe {
                let gh0 = *grad_hess.get_unchecked(row0);
                let gh1 = *grad_hess.get_unchecked(row1);   
                let packed0 = *bundle.packed_bins.get_unchecked(row0);
                let packed1 = *bundle.packed_bins.get_unchecked(row1);

                let bin0_0 = (packed0 & 0xFF) as usize;
                let bin1_0 = ((packed0 >> 8) & 0xFF) as usize;
                let bin2_0 = ((packed0 >> 16) & 0xFF) as usize;
                let bin3_0 = (packed0 >> 24) as usize;

                let bin0_1 = (packed1 & 0xFF) as usize;
                let bin1_1 = ((packed1 >> 8) & 0xFF) as usize;
                let bin2_1 = ((packed1 >> 16) & 0xFF) as usize;
                let bin3_1 = (packed1 >> 24) as usize;

                (*p0.add(bin0_0)).add(gh0);
                (*p1.add(bin1_0)).add(gh0);
                (*p2.add(bin2_0)).add(gh0);
                (*p3.add(bin3_0)).add(gh0);
                
                (*p0.add(bin0_1)).add(gh1);
                (*p1.add(bin1_1)).add(gh1);
                (*p2.add(bin2_1)).add(gh1);
                (*p3.add(bin3_1)).add(gh1);

            }
        }

        // clean up the remaining row if the length was odd
        for &row in chunks.remainder() {
            let r = row as usize;
            unsafe {
                let gh = *grad_hess.get_unchecked(r);
                let packed = *bundle.packed_bins.get_unchecked(r);

                let b0 = (packed & 0xFF) as usize;
                let b1 = ((packed >> 8) & 0xFF) as usize;
                let b2 = ((packed >> 16) & 0xFF) as usize;
                let b3 = (packed >> 24) as usize;

                (*p0.add(b0)).add(gh);
                (*p1.add(b1)).add(gh);
                (*p2.add(b2)).add(gh);
                (*p3.add(b3)).add(gh);
            }
        }

        let mut results = vec![
            Self { bins: h0, is_categorical: bundle.is_categorical.get(0).copied().unwrap_or(false) },
            Self { bins: h1, is_categorical: bundle.is_categorical.get(1).copied().unwrap_or(false) },
            Self { bins: h2, is_categorical: bundle.is_categorical.get(2).copied().unwrap_or(false) },
            Self { bins: h3, is_categorical: bundle.is_categorical.get(3).copied().unwrap_or(false) },
        ];
        results.truncate(bundle.count);
        results
    }

    /// Write `parent - child` into `self`. In-place operation.
    pub fn subtract(&mut self, parent: &Histogram, child: &Histogram) {
        debug_assert_eq!(self.bins.len(), parent.bins.len());
        debug_assert_eq!(self.bins.len(), child.bins.len());

        for (s, (p, c)) in self.bins.iter_mut().zip(parent.bins.iter().zip(child.bins.iter())) {
            s.sum_gradients = p.sum_gradients - c.sum_gradients;
            s.sum_hessians = p.sum_hessians - c.sum_hessians;
        }
    }

    /// Find the best numeric split by scanning bins left to right.
    pub fn find_best_numeric_split(
        &self,
        total_gradient: f64,
        total_hessian: f64,
        parent_score: f64,
        parameters: &Parameters,
    ) -> Option<SplitInfo> {
        let sentinel_bin = &self.bins.last().unwrap();

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
            if self.bins.len() < 3 { return None; }
            // Safe due to the check above.
            for t in 0..self.bins.len() - 2 {
                let bin = &self.bins[t];
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
            if self.bins.len() < 2 { return None; }
            // One additional split to check compared to above: All missing values go
            // left, everything else goes right.
            for t in 0..self.bins.len() - 1 {
                // first compute score for missing_goes_left = false.
                left_gradient += self.bins[t].sum_gradients;
                left_hessian += self.bins[t].sum_hessians;

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

        // Only ever split with positive gain.
        if best_score <= parent_score { return None; }

        Some(SplitInfo {
            gain: best_score - parent_score,
            missing_goes_left: best_missing_goes_left,
            threshold: Threshold::Numeric(best_threshold as u32),
            feature_index: 0, // to be filled in by caller
        })
    }

    /// Find the best categorical split by sorting bins by gradient ratio.
    /// 
    /// Instead of checking all 2^num_bins subsets, we first sort categories by their
    /// gradient/hessian ratio. Then we only need to check splits between sorted
    /// categories. This is exact according to Fisher, W. D. (1958).
    pub fn find_best_categorical_split(
        &self,
        total_gradient: f64,
        total_hessian: f64,
        parent_score: f64,
        parameters: &Parameters,
    ) -> Option<SplitInfo> {
        let num_bins = self.bins.len();

        // We treat missing values as just another category.
        let mut categorical_order: Vec<(f64, usize)> = Vec::with_capacity(num_bins);
        for k in 0..num_bins {
            let bin = &self.bins[k];
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
            let bin = &self.bins[categorical_order[t].1];
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

        if best_score <= parent_score { return None; }

        let mut best_missing_goes_left = best_majority_goes_left;
        let mut goes_left = vec![best_majority_goes_left; num_bins - 1];
    
        for (i, &(_, k)) in categorical_order.iter().enumerate() {
            let is_left = i <= best_threshold;
            
            if k == num_bins - 1 {
                best_missing_goes_left = is_left;
            } else {
                goes_left[k] = is_left;
            }
        }

        Some(SplitInfo {
            gain: best_score - parent_score,
            missing_goes_left: best_missing_goes_left,
            threshold: Threshold::Categorical(goes_left),
            feature_index: 0, // to be filled in by caller
        })
    }
}

/// Score of a leaf node used for gain calculation.
///
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
    parameters: &Parameters,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::FeatureBundle;
    use crate::parameters::Parameters;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-7, "{a} != {b}");
    }

    fn p() -> Parameters {
        Parameters { min_sum_hessian_in_leaf: 0.0, ..Parameters::default() }
    }

    fn make_bundle(bins: &[u8], num_bins: usize, is_categorical: bool) -> FeatureBundle {
        FeatureBundle {
            packed_bins: bins.iter().map(|&b| b as u32).collect(),
            feature_indices: vec![0],
            num_bins: vec![num_bins],
            is_categorical: vec![is_categorical],
            count: 1,
        }
    }

    fn make_gh(gradients: &[f64], hessians: &[f64]) -> Vec<[f32; 2]> {
        gradients.iter().zip(hessians).map(|(&g, &h)| [g as f32, h as f32]).collect()
    }

    #[test]
    fn test_histogram_build_and_subtract() {
        let num_bins = 4; // 3 regular bins (0,1,2) + 1 sentinel (3)
        let feature_column = vec![0u8, 1, 0, 2];
        let gh = make_gh(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4]);
        let bundle = make_bundle(&feature_column, num_bins, false);
        let row_indices: Vec<u32> = vec![0, 1, 2, 3];

        let parent = Histogram::build(&bundle, &gh, &row_indices).remove(0);
        assert_approx_eq(parent.bins[0].sum_gradients, 4.0);
        assert_approx_eq(parent.bins[0].sum_hessians, 2.0);
        assert_approx_eq(parent.bins[1].sum_gradients, 2.0);
        assert_approx_eq(parent.bins[2].sum_gradients, 4.0);
        assert_approx_eq(parent.bins[3].sum_hessians, 0.0); // sentinel empty

        // rows 0 and 2 both fall into bin 0; right = parent - left contains only rows 1 and 3
        let right = Histogram::build(&bundle, &gh, &[1u32, 3]).remove(0);
        assert_approx_eq(right.bins[0].sum_hessians, 0.0);
        assert_approx_eq(right.bins[1].sum_gradients, 2.0);
        assert_approx_eq(right.bins[2].sum_gradients, 4.0);
    }

    #[test]
    fn test_find_best_numeric_split() {
        // Bins 0+1 (G=-20, H=20) vs bin 2 (G=20, H=10): gain = 400/20 + 400/10 = 60.
        // No missings: missing_goes_left = left_count(20) > right_count(10).
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0 },
        ], is_categorical: false };
        let split = hist.find_best_numeric_split(0.0, 30.0, 0.0, &parameters).unwrap();
        assert!(matches!(split.threshold, Threshold::Numeric(1)));
        assert_approx_eq(split.gain, 60.0);
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_find_best_numeric_no_split_when_uniform() {
        // Uniform gradients: every split yields the same score as the parent.
        let bundle = make_bundle(&[0u8, 1, 2], 3, false);
        let gh = make_gh(&[1.0, 1.0, 1.0], &[1.0; 3]);
        let hist = Histogram::build(&bundle, &gh, &[0u32, 1, 2]).remove(0);
        assert!(hist.find_best_numeric_split(3.0, 3.0, 3.0, &p()).is_none());
    }

    #[test]
    fn test_numeric_split_with_missings() {
        // missing_goes_left: left G=-20,H=20 → 20; right G=20,H=10 → 40; gain=60.
        // missing_goes_right: left G=-10,H=10 → 10; right G=10,H=20 → 5; gain=15.
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 }, // sentinel
        ], is_categorical: false };
        let split = hist.find_best_numeric_split(0.0, 30.0, 0.0, &parameters).unwrap();
        assert_approx_eq(split.gain, 60.0);
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_find_best_categorical_split() {
        // Bins 0 and 2 share ratio -1, bin 1 has ratio +2.
        // Fisher sorting groups 0 and 2 into left prefix → gain = 400/20 + 400/10 = 60.
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0 },
        ], is_categorical: true };
        let split = hist.find_best_categorical_split(0.0, 30.0, 0.0, &parameters).unwrap();
        assert_approx_eq(split.gain, 60.0);
        match &split.threshold {
            Threshold::Categorical(goes_left) => {
                assert_eq!(goes_left.len(), 3);
                assert!(goes_left[0]);
                assert!(goes_left[2]);
                assert!(!goes_left[1]);
            }
            _ => panic!("expected categorical threshold"),
        }
    }

    #[test]
    fn test_categorical_missing_goes_left() {
        // Sentinel has strongly negative gradient: optimal to send missings left.
        let feature_bins: Vec<u8> = (0..10).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let grads: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { -5.0 }).collect();
        let gh = make_gh(&grads, &vec![1.0; 10]);
        let bundle = make_bundle(&feature_bins, 2, true); // 1 category + 1 sentinel
        let hist = Histogram::build(&bundle, &gh, &(0..10u32).collect::<Vec<_>>()).remove(0);
        let (g, h) = hist.bins.iter().fold((0.0, 0.0), |(g, h), b| (g + b.sum_gradients, h + b.sum_hessians));
        let split = hist.find_best_categorical_split(g, h, 0.0, &p()).unwrap();
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_categorical_no_missings_missing_goes_to_majority() {
        // No missings (sentinel empty). Left side has more hessian → missing_goes_left = true.
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 20.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0 }, // sentinel empty
        ], is_categorical: true };
        let split = hist.find_best_categorical_split(10.0, 30.0, 0.0, &p()).unwrap();
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_categorical_zero_hessian_bin_routed_to_majority() {
        // Bin 2 has zero hessian: should go to the majority (heavier) side.
        // Left (bin 0, h=15) is heavier than right (bin 1, h=5), so bin 2 goes left.
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 15.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians:  5.0 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0 }, // zero hessian
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0 }, // sentinel
        ], is_categorical: true };
        let split = hist.find_best_categorical_split(10.0, 20.0, 0.0, &p()).unwrap();
        match &split.threshold {
            Threshold::Categorical(goes_left) => {
                assert!(goes_left[0], "bin 0 goes left");
                assert!(!goes_left[1], "bin 1 goes right");
                assert!(goes_left[2], "zero-hessian bin goes to majority (left, h=15 > h=5)");
            }
            _ => panic!("expected categorical threshold"),
        }
    }

    #[test]
    fn test_categorical_missing_placed_by_fisher_sort() {
        // Sentinel has strongly positive gradient: Fisher sort puts it on the right side.
        // bin 0: ratio=-2, goes left. sentinel: ratio=1.0, bin 1: ratio=1.33, both go right.
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians:  5.0 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 15.0 },
            HistogramBin { sum_gradients:  10.0, sum_hessians: 10.0 }, // sentinel (missings)
        ], is_categorical: true };
        let split = hist.find_best_categorical_split(20.0, 30.0, 0.0, &p()).unwrap();
        assert!(!split.missing_goes_left);
        match &split.threshold {
            Threshold::Categorical(goes_left) => {
                assert!(goes_left[0], "bin 0 goes left");
                assert!(!goes_left[1], "bin 1 goes right");
            }
            _ => panic!("expected categorical threshold"),
        }
    }

}

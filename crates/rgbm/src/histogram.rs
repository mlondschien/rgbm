//! Histogram-based gradient and hessian accumulation, and best-split finding.

use crate::dataset::FeatureBundle;
use crate::parameters::Parameters;
use crate::utils::calculate_score;


#[derive(Clone, Default, Debug)]
#[repr(C)]  // group fields into 16 bytes for optimal SIMD processing
pub struct HistogramBin {
    pub sum_gradients: f64,
    pub sum_hessians: f64,
}

impl HistogramBin {
    #[inline(always)]
    pub fn add(&mut self, gh: [f64; 2]) {
        self.sum_gradients += gh[0];
        self.sum_hessians += gh[1];
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
    pub left_gradient: f64,
    pub right_gradient: f64,
    pub left_hessian: f64,
    pub right_hessian: f64,
    pub left_score: f64,
    pub right_score: f64,
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
    /// SAFETY: bin values are in 0..num_bins[i] (sentinel is at num_bins[i] - 1),
    /// histograms are sized num_bins[i].
    pub fn build(bundle: &FeatureBundle, grad_hess: &[[f64; 2]], row_indices: &[u32]) -> Vec<Self> {
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

        for &row in row_indices {
            let row = row as usize;
            unsafe {
                let gh = *grad_hess.get_unchecked(row);
                let packed = *bundle.packed_bins.get_unchecked(row);

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
        let num_bins = self.bins.len().saturating_sub(1);
        if num_bins == 0 { return None; }

        let sentinel = &self.bins[num_bins];  // bin containing missings

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;

        let mut cumsum_gradients = Vec::with_capacity(num_bins);
        let mut cumsum_hessians = Vec::with_capacity(num_bins);

        // Pass 1 through the data: Compute cumulative sums.
        for bin in &self.bins[..num_bins] {
            left_gradient += bin.sum_gradients;
            left_hessian += bin.sum_hessians;

            cumsum_gradients.push(left_gradient);
            cumsum_hessians.push(left_hessian);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;
        let mut best_missing_goes_left = false;

        // Code duplication: Two loops with the same gain calculation but with the
        // branch on the outside
        if sentinel.sum_hessians == 0.0 {
            // Pass 2: SIMD-friendly gain calculation. The goal is to have no loop-
            // carried dependencies so the compiler can perfectly unroll and vectorize.
            // Stop at num_bins - 1: threshold t means bins 0..=t go left, so right
            // must have at least one non-sentinel bin (t = num_bins - 1 is degenerate).
            for t in 0..num_bins - 1 {
                let left_gradient = cumsum_gradients[t];
                let left_hessian = cumsum_hessians[t];
                let right_hessian = total_hessian - left_hessian;
                let right_gradient = total_gradient - left_gradient;

                // gain = score - parent_score
                // We always compute the score, even if leaf constraints
                // (min_sum_hessian_in_leaf) are not met. This allows the compiler to
                // perfectly unroll without any branching issues.
                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);

                // Use & instead of && to avoid branching.
                let leaf_constraint = (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
                let score = if leaf_constraint { score } else { f64::NEG_INFINITY };
                if score > best_score {
                    best_score = score;
                    best_threshold = t;
                    // If no missings at train time but some at test time, send missings
                    // to the larger side of the split.
                    best_missing_goes_left = left_hessian > right_hessian;
                }
            }
        } else {
            for t in 0..num_bins.saturating_sub(1) {
                // Same code as above. First compute score for missing_goes_left = false.
                let left_gradient = cumsum_gradients[t];
                let left_hessian = cumsum_hessians[t];
                let right_hessian = total_hessian - left_hessian;
                let right_gradient = total_gradient - left_gradient;

                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
                let leaf_constraint = (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
                let score = if leaf_constraint { score } else { f64::NEG_INFINITY };

                if score > best_score {
                    best_score = score;
                    best_threshold = t;
                    best_missing_goes_left = false;
                }

                // Now compute score for missing_goes_left = true.
                // new assignments so compiler knows there's no dependencies
                let left_hessian = left_hessian + sentinel.sum_hessians;
                let left_gradient = left_gradient + sentinel.sum_gradients;
                let right_hessian = right_hessian - sentinel.sum_hessians;
                let right_gradient = right_gradient - sentinel.sum_gradients;

                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
                let leaf_constraint = (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
                let score = if leaf_constraint { score } else { f64::NEG_INFINITY };

                if score > best_score {
                    best_score = score;
                    best_threshold = t;
                    best_missing_goes_left = true;
                }
            }
        }

        // Only ever split with positive gain.
        if best_score <= parent_score { return None; }

        let left_gradient = cumsum_gradients[best_threshold]
            + if best_missing_goes_left { sentinel.sum_gradients } else { 0.0 };
        let left_hessian  = cumsum_hessians[best_threshold]
            + if best_missing_goes_left { sentinel.sum_hessians } else { 0.0 };
        let right_gradient = total_gradient - left_gradient;
        let right_hessian  = total_hessian - left_hessian;

        Some(SplitInfo {
            gain: best_score - parent_score,
            missing_goes_left: best_missing_goes_left,
            threshold: Threshold::Numeric(best_threshold as u32),
            left_gradient,
            right_gradient,
            left_hessian,
            right_hessian,
            left_score: calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2),
            right_score: calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2),
        })
    }

    /// Find the best categorical split by sorting bins by gradient ratio.
    pub fn find_best_categorical_split(
        &self,
        total_gradient: f64,
        total_hessian: f64,
        parent_score: f64,
        parameters: &Parameters,
    ) -> Option<SplitInfo> {
        let num_bins = self.bins.len().saturating_sub(1);
        if num_bins == 0 { return None; }

        // Instead of checking all 2^num_bins subsets, we first sort categories by their
        // gradient/hessian ratio. Then we only need to check splits between sorted
        // categories. This is exact according to Fisher, W. D. (1958).

        // PASS 1: Filter active bins AND the sentinel bin (index == num_bins).
        // By using `0..=num_bins`, we treat missing values as just another category.
        let mut categorical_order: Vec<(f64, usize)> = Vec::with_capacity(num_bins + 1);
        for k in 0..=num_bins {
            let bin = &self.bins[k];
            if bin.sum_hessians > 0.0 {
                let ratio = bin.sum_gradients / (bin.sum_hessians + parameters.lambda_l2);
                categorical_order.push((ratio, k));
            }
        }

        if categorical_order.is_empty() { return None; }

        // Sort categories to find the optimal contiguous binary partition
        categorical_order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let num_active_bins = categorical_order.len();

        let mut cumsum_gradients = Vec::with_capacity(num_active_bins);
        let mut cumsum_hessians = Vec::with_capacity(num_active_bins);

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;

        // PASS 2: Compute cumsums over the SORTED categorical order
        for &(_, k) in &categorical_order {
            let bin = &self.bins[k];
            left_gradient += bin.sum_gradients;
            left_hessian += bin.sum_hessians;

            cumsum_gradients.push(left_gradient);
            cumsum_hessians.push(left_hessian);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;

        // PASS 3: SIMD-friendly gain calculation.
        for t in 0..num_active_bins.saturating_sub(1){
            let left_gradient = cumsum_gradients[t];
            let left_hessian = cumsum_hessians[t];

            let right_hessian = total_hessian - left_hessian;
            let right_gradient = total_gradient - left_gradient;

            let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
            let leaf_constraint = (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
            let score = if leaf_constraint { score } else { f64::NEG_INFINITY };

            if score > best_score {
                best_score = score;
                best_threshold = t;
            }
        }

        if best_score <= parent_score { return None; }

        let mut goes_left = vec![false; num_bins];
        let mut missing_goes_left = false;

        for &(_, k) in &categorical_order[..best_threshold + 1] {
            if k == num_bins {
                missing_goes_left = true;
            } else {
                goes_left[k] = true;
            }
        }

        let left_gradient = cumsum_gradients[best_threshold];
        let left_hessian = cumsum_hessians[best_threshold];
        let right_gradient = total_gradient - left_gradient;
        let right_hessian = total_hessian - left_hessian;

        Some(SplitInfo {
            gain: best_score - parent_score,
            missing_goes_left,
            threshold: Threshold::Categorical(goes_left),
            left_gradient,
            right_gradient,
            left_hessian,
            right_hessian,
            left_score: calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2),
            right_score: calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2),
        })
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

    fn make_gh(gradients: &[f64], hessians: &[f64]) -> Vec<[f64; 2]> {
        gradients.iter().zip(hessians).map(|(&g, &h)| [g, h]).collect()
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
        assert_approx_eq(split.left_score, 20.0);
        assert_approx_eq(split.right_score, 40.0);
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

}

//! Histogram-based gradient and hessian accumulation, and best-split finding.

use crate::parameters::Parameters;
use crate::utils::calculate_score;

#[derive(Clone, Default, Debug)]
pub struct HistogramBin {
    pub sum_gradients: f64,
    pub sum_hessians: f64,
    pub count: u32,
}

pub struct Scratch {  // reuse memory in split finding
    cumsum_gradients: Vec<f64>,
    cumsum_hessians: Vec<f64>,
    cumsum_counts: Vec<u32>,
    // (gradient ratio, bin index) pairs for categorical splits
    categorical_order: Vec<(f64, usize)>,
}

impl Scratch {
    pub fn new(num_bins: usize) -> Self {
        Self {
            cumsum_gradients: Vec::with_capacity(num_bins),
            cumsum_hessians: Vec::with_capacity(num_bins),
            cumsum_counts: Vec::with_capacity(num_bins),
            categorical_order: Vec::with_capacity(num_bins),
        }
    }

    pub fn clear(&mut self) {
        self.cumsum_gradients.clear();
        self.cumsum_hessians.clear();
        self.cumsum_counts.clear();
        self.categorical_order.clear();
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
///
/// `bins.len() == num_bins + 1`; the last entry is the sentinel bin for missings.
pub struct Histogram {
    pub bins: Vec<HistogramBin>,
}

impl Histogram {
    /// Build a histogram by accumulating gradients and hessians for `row_indices`.
    /// This is essentially a group-by-sum operation. Histograms are built for each
    /// feature and for each tree node during training. They aggregate over all rows
    /// and thus are a bottleneck. We use unsafe code for maximum performance. 
    pub fn build(
        feature_column: &[u16],
        gradients: &[f64],
        hessians: &[f64],
        // using u32 for row indices saves some memory that gets copied around a lot.
        // u32 ~ 4 billion rows should be plenty.
        row_indices: &[u32],
        num_bins: usize,
    ) -> Self {
        let mut bins = vec![HistogramBin::default(); num_bins + 1]; 

        for &row in row_indices {
            let row = row as usize;
            // SAFETY: row < feature_column.len() (row_indices are valid row indices),
            // bin_idx <= num_bins (guaranteed by the binners), bins.len() == num_bins + 1.
            unsafe {
                // Use get_unchecked and get_unchecked_mut to avoid (costly) bounds
                // checks. This loop is executed very often.
                let bin_idx = *feature_column.get_unchecked(row) as usize;
                debug_assert!(bin_idx < bins.len(), "bin index {bin_idx} out of bounds");
                let bin = bins.get_unchecked_mut(bin_idx);
                // Safe as the binners guarantee that `feature_column` values are
                // `<= num_bins`.
                bin.sum_gradients += *gradients.get_unchecked(row);
                bin.sum_hessians += *hessians.get_unchecked(row);
                bin.count += 1;
            }
        }

        Self { bins }
    }

    /// Write `parent - child` into `self`. In-place operation.
    pub fn subtract(&mut self, parent: &Histogram, child: &Histogram) {
        debug_assert_eq!(self.bins.len(), parent.bins.len());
        debug_assert_eq!(self.bins.len(), child.bins.len());

        for (s, (p, c)) in self.bins.iter_mut().zip(parent.bins.iter().zip(child.bins.iter())) {
            s.sum_gradients = p.sum_gradients - c.sum_gradients;
            s.sum_hessians = p.sum_hessians - c.sum_hessians;
            s.count = p.count - c.count;
        }
    }

    /// Find the best numeric split by scanning bins left to right.
    pub fn find_best_numeric_split(
        &self,
        total_gradient: f64,
        total_hessian: f64,
        total_count: u32,
        parent_score: f64,
        parameters: &Parameters,
        scratch: &mut Scratch,
    ) -> Option<SplitInfo> {
        let num_bins = self.bins.len().saturating_sub(1);
        if num_bins == 0 { return None; }

        let sentinel = &self.bins[num_bins];  // bin containing missings

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;
        let mut left_count = 0u32;

        scratch.clear();

        // Let's hope the reserve call helps the compiler remove bounds checks in the 
        // loop below.
        scratch.cumsum_counts.reserve(num_bins);
        scratch.cumsum_gradients.reserve(num_bins);
        scratch.cumsum_hessians.reserve(num_bins);

        // Pass 1 through the data: Compute cumulative sums.
        for bin in &self.bins[..num_bins] {
            left_gradient += bin.sum_gradients;
            left_hessian += bin.sum_hessians;
            left_count += bin.count;

            scratch.cumsum_gradients.push(left_gradient);
            scratch.cumsum_hessians.push(left_hessian);
            scratch.cumsum_counts.push(left_count);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;
        let mut best_missing_goes_left = false;

        // Code duplication: Two loops with the same gain calculation but with the
        // branch on the outside
        if sentinel.count == 0 {
            // Pass 2: SIMD-friendly gain calculation. The goal is to have no loop-
            // carried dependencies so the compiler can perfectly unroll and vectorize.
            for t in 0..num_bins {
                let left_gradient = scratch.cumsum_gradients[t];
                let left_hessian = scratch.cumsum_hessians[t];
                let left_count = scratch.cumsum_counts[t];

                let right_count = total_count - left_count;
                let right_hessian = total_hessian - left_hessian;
                let right_gradient = total_gradient - left_gradient;
                
                // gain = score - parent_score
                // We always compute the score, even if leaf constraints
                // (min_data_in_leaf, min_sum_hessian_in_leaf) are not met. This allows
                // the compiler to perfectly unroll without any branching issues. One
                // could also compute the min_idx and max_idx based on leaf constraints.
                // Unsure if that would be worth it.
                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
                
                // Use & instead of && to avoid branching.
                let leaf_constraint = (left_count >= parameters.min_data_in_leaf as u32) & (right_count >= parameters.min_data_in_leaf as u32) & (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
                // Instead of `if leaf_constraint & (score > best_score)` below, make
                // the if/else branch as simple as possible to make it more likely the
                // compiler autovectorizes.
                let score = if leaf_constraint { score } else { f64::NEG_INFINITY };
                if score > best_score {
                    best_score = score;
                    best_threshold = t;
                    // If no missings at train time but some at test time, send missings
                    // to the larger side of the split.
                    best_missing_goes_left = left_count > right_count;
                }
            }
        } else {
            for t in 0..num_bins {
                // Same code as above. First compute score for missing_goes_left = false.
                // Note that total_count includes the missing values bin.
                let left_gradient = scratch.cumsum_gradients[t];
                let left_hessian = scratch.cumsum_hessians[t];
                let left_count = scratch.cumsum_counts[t];

                let right_count = total_count - left_count;
                let right_hessian = total_hessian - left_hessian;
                let right_gradient = total_gradient - left_gradient;
                
                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
                let leaf_constraint = (left_count >= parameters.min_data_in_leaf as u32) & (right_count >= parameters.min_data_in_leaf as u32) & (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
                let score = if leaf_constraint { score } else { f64::NEG_INFINITY };
    
                if score > best_score {
                    best_score = score;
                    best_threshold = t;
                    best_missing_goes_left = false;
                }

                // Now compute score for missing_goes_left = true.
                // new assignments so compiler knows there's no depdendencies
                let left_count = left_count + sentinel.count;
                let left_hessian = left_hessian + sentinel.sum_hessians;
                let left_gradient = left_gradient + sentinel.sum_gradients;

                let right_count = right_count - sentinel.count;
                let right_hessian = right_hessian - sentinel.sum_hessians;
                let right_gradient = right_gradient - sentinel.sum_gradients;

                let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
                let leaf_constraint = (left_count >= parameters.min_data_in_leaf as u32) & (right_count >= parameters.min_data_in_leaf as u32) & (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
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

        let left_gradient = scratch.cumsum_gradients[best_threshold] 
            + if best_missing_goes_left { sentinel.sum_gradients } else { 0.0 };
        let left_hessian  = scratch.cumsum_hessians[best_threshold] 
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
        total_count: u32,
        parent_score: f64,
        parameters: &Parameters,
        scratch: &mut Scratch,
    ) -> Option<SplitInfo> {
        let num_bins = self.bins.len().saturating_sub(1);
        if num_bins == 0 { return None; }

        scratch.clear();
        
        // Instead of checking all 2^num_bins subsets, we first sort categories by their
        // gradient/hessian ratio. Then we only need to check splits between sorted
        // categories. This is exact according to Fisher, W. D. (1958).

        // PASS 1: Filter active bins AND the sentinel bin (index == num_bins).
        // By using `0..=num_bins`, we treat missing values as just another category.
        for k in 0..=num_bins {
            let bin = &self.bins[k];
            if bin.count > 0 {
                let ratio = bin.sum_gradients / (bin.sum_hessians + parameters.lambda_l2);
                scratch.categorical_order.push((ratio, k));
            }
        }

        if scratch.categorical_order.is_empty() { return None; }

        // Sort categories to find the optimal contiguous binary partition
        scratch.categorical_order.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let num_active_bins = scratch.categorical_order.len();

        scratch.cumsum_counts.reserve(num_active_bins);
        scratch.cumsum_gradients.reserve(num_active_bins);
        scratch.cumsum_hessians.reserve(num_active_bins);

        let mut left_gradient = 0.0;
        let mut left_hessian = 0.0;
        let mut left_count = 0u32;

        // PASS 2: Compute cumsums over the SORTED categorical order
        for &(_, k) in &scratch.categorical_order {
            let bin = &self.bins[k];
            left_gradient += bin.sum_gradients;
            left_hessian += bin.sum_hessians;
            left_count += bin.count;

            scratch.cumsum_gradients.push(left_gradient);
            scratch.cumsum_hessians.push(left_hessian);
            scratch.cumsum_counts.push(left_count);
        }

        let mut best_score = f64::NEG_INFINITY;
        let mut best_threshold = 0usize;

        // PASS 3: SIMD-friendly gain calculation. 
        for t in 0..num_active_bins {
            let left_gradient = scratch.cumsum_gradients[t];
            let left_hessian = scratch.cumsum_hessians[t];
            let left_count = scratch.cumsum_counts[t];

            let right_count = total_count - left_count;
            let right_hessian = total_hessian - left_hessian;
            let right_gradient = total_gradient - left_gradient;
            
            let score = calculate_score(left_gradient, left_hessian, parameters.lambda_l1, parameters.lambda_l2) + calculate_score(right_gradient, right_hessian, parameters.lambda_l1, parameters.lambda_l2);
            let leaf_constraint = (left_count >= parameters.min_data_in_leaf as u32) & (right_count >= parameters.min_data_in_leaf as u32) & (left_hessian >= parameters.min_sum_hessian_in_leaf) & (right_hessian >= parameters.min_sum_hessian_in_leaf);
            let score = if leaf_constraint { score } else { f64::NEG_INFINITY };
    
            if score > best_score {
                best_score = score;
                best_threshold = t;
            }
        }

        if best_score <= parent_score { return None; }

        let mut goes_left = vec![false; num_bins];
        let mut missing_goes_left = false;

        for &(_, k) in &scratch.categorical_order[..best_threshold + 1] {
            if k == num_bins {
                missing_goes_left = true;
            } else {
                goes_left[k] = true;
            }
        }

        let left_gradient = scratch.cumsum_gradients[best_threshold];
        let left_hessian = scratch.cumsum_hessians[best_threshold];
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
    use crate::parameters::Parameters;

    fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-7, "{a} != {b}");
    }

    fn p() -> Parameters {
        Parameters { min_data_in_leaf: 1, min_sum_hessian_in_leaf: 0.0, ..Parameters::default() }
    }

    #[test]
    fn test_histogram_build_and_subtract() {
        let num_bins = 3;
        let feature_column = vec![0u16, 1, 0, 2];
        let gradients = vec![1.0, 2.0, 3.0, 4.0];
        let hessians = vec![1.0; 4];
        let row_indices: Vec<u32> = vec![0, 1, 2, 3];

        let parent = Histogram::build(&feature_column, &gradients, &hessians, &row_indices, num_bins);
        assert_eq!(parent.bins[0].count, 2);
        assert_approx_eq(parent.bins[0].sum_gradients, 4.0);
        assert_eq!(parent.bins[1].count, 1);
        assert_approx_eq(parent.bins[1].sum_gradients, 2.0);
        assert_eq!(parent.bins[2].count, 1);
        assert_approx_eq(parent.bins[2].sum_gradients, 4.0);
        assert_eq!(parent.bins[3].count, 0); // sentinel empty

        // rows 0 and 2 both fall into bin 0; right = parent - left contains only rows 1 and 3
        let left = Histogram::build(&feature_column, &gradients, &hessians, &[0u32, 2], num_bins);
        let mut right = Histogram { bins: vec![HistogramBin::default(); 4] };
        right.subtract(&parent, &left);
        assert_eq!(right.bins[0].count, 0);
        assert_eq!(right.bins[1].count, 1);
        assert_eq!(right.bins[2].count, 1);
    }

    #[test]
    fn test_find_best_numeric_split() {
        // Bins 0+1 (G=-20, H=20) vs bin 2 (G=20, H=10): gain = 400/20 + 400/10 = 60.
        // No missings: missing_goes_left = left_count(20) > right_count(10).
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0, count:  0 },
        ]};
        let split = hist.find_best_numeric_split(0.0, 30.0, 30, 0.0, &parameters, &mut Scratch::new(3)).unwrap();
        assert!(matches!(split.threshold, Threshold::Numeric(1)));
        assert_approx_eq(split.gain, 60.0);
        assert_approx_eq(split.left_score, 20.0);
        assert_approx_eq(split.right_score, 40.0);
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_find_best_numeric_no_split_when_uniform() {
        // Uniform gradients: every split yields the same score as the parent.
        let feature_bins = vec![0u16, 1, 2];
        let grads = vec![1.0, 1.0, 1.0];
        let hess = vec![1.0; 3];
        let hist = Histogram::build(&feature_bins, &grads, &hess, &[0u32, 1, 2], 3);
        assert!(hist.find_best_numeric_split(3.0, 3.0, 3, 3.0, &p(), &mut Scratch::new(3)).is_none());
    }

    #[test]
    fn test_numeric_split_with_missings() {
        // missing_goes_left: left G=-20,H=20 → 20; right G=20,H=10 → 40; gain=60.
        // missing_goes_right: left G=-10,H=10 → 10; right G=10,H=20 → 5; gain=15.
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 }, // sentinel
        ]};
        let split = hist.find_best_numeric_split(0.0, 30.0, 30, 0.0, &parameters, &mut Scratch::new(2)).unwrap();
        assert_approx_eq(split.gain, 60.0);
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_find_best_categorical_split() {
        // Bins 0 and 2 share ratio -1, bin 1 has ratio +2.
        // Fisher sorting groups 0 and 2 into left prefix → gain = 400/20 + 400/10 = 60.
        let parameters = p();
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:  20.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0, count:  0 },
        ]};
        let split = hist.find_best_categorical_split(0.0, 30.0, 30, 0.0, &parameters, &mut Scratch::new(3)).unwrap();
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
        let feature_bins: Vec<u16> = (0..10).map(|i| if i < 5 { 0 } else { 1 }).collect();
        let grads: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { -5.0 }).collect();
        let hess = vec![1.0; 10];
        let hist = Histogram::build(&feature_bins, &grads, &hess, &(0..10u32).collect::<Vec<_>>(), 1);
        let (g, h, c) = hist.bins.iter().fold((0.0, 0.0, 0u32), |(g, h, c), b| (g + b.sum_gradients, h + b.sum_hessians, c + b.count));
        let split = hist.find_best_categorical_split(g, h, c, 0.0, &p(), &mut Scratch::new(1)).unwrap();
        assert!(split.missing_goes_left);
    }

    #[test]
    fn test_leaf_constraints() {
        // Mathematically a split exists, but both children would have only 10 rows < min_data_in_leaf=15.
        let parameters = Parameters { min_data_in_leaf: 15, ..p() };
        let hist = Histogram { bins: vec![
            HistogramBin { sum_gradients: -10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:  10.0, sum_hessians: 10.0, count: 10 },
            HistogramBin { sum_gradients:   0.0, sum_hessians:  0.0, count:  0 },
        ]};
        assert!(hist.find_best_numeric_split(0.0, 20.0, 20, 0.0, &parameters, &mut Scratch::new(2)).is_none());
    }
}

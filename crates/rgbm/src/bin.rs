//! Feature binning: converts continuous f64 values and categorical strings to bin indices.

use ahash::AHashMap;

use arrow::array::{Array, AsArray};
use arrow::datatypes::{DataType, Float64Type};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::IteratorRandom;


pub enum BinnerKind {
    /// Upper bounds for each bin. The last entry is always +inf.
    /// A value `x` maps to bin `i` if `i` is maximal s.t. `x <= upper_bounds[i]`.
    Numerical(Vec<f64>),
    /// Maps each category string to its bin index (position in the training dictionary).
    Categorical(AHashMap<String, u8>),
}

/// Stores both the bin mapping rules and the pre-computed bin indices for the training data.
/// Nulls, NaNs, and unknown categories map to a sentinel bin at index `num_bins()`.
pub struct FeatureBinner {
    pub bins: Vec<u8>,
    pub kind: BinnerKind,
}

impl FeatureBinner {
    /// Build a `FeatureBinner` from an Arrow array. Dispatches on array type:
    /// Float64 -> numerical binning, Dictionary -> categorical binning.
    pub fn from_array(array: &dyn Array, num_bins: usize, min_data_in_bin: usize) -> Self {
        match array.data_type() {
            DataType::Float64 => {
                assert!(num_bins > 0, "num_bins must be at least 1, got {num_bins}");
                let values = array.as_primitive::<Float64Type>();

                // todo: use booster's seed once we have it
                const MAX_SAMPLE: usize = 200_000;  // same as LGBM
                let mut rng = StdRng::seed_from_u64(0);
                let mut valid = values.iter().flatten().filter(|x| !x.is_nan())
                    .choose_multiple(&mut rng, MAX_SAMPLE);
                valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                let upper_bounds = greedy_find_bins(&valid, num_bins, min_data_in_bin);
                let sentinel = upper_bounds.len() as u8;
                let raw_values = values.values();
                let mut bins = Vec::with_capacity(raw_values.len());
                if let Some(nulls) = values.nulls() {
                    for (i, &x) in raw_values.iter().enumerate() {
                        bins.push(if nulls.is_null(i) || x.is_nan() { sentinel } else { upper_bounds.partition_point(|&b| b < x) as u8 });
                    }
                } else {
                    for &x in raw_values {
                        bins.push(if x.is_nan() { sentinel } else { upper_bounds.partition_point(|&b| b < x) as u8 });
                    }
                }

                Self { bins, kind: BinnerKind::Numerical(upper_bounds) }
            }
            DataType::Dictionary(_, _) => {
                let dict = array.as_any_dictionary();
                let dict_values = dict.values().as_string::<i32>();

                // todo: implement multi-category splits (LightGBM-style 4-byte bitset per bin)
                assert!(dict_values.len() <= num_bins, "categorical feature has {} categories, max is {num_bins}; reduce the number of categories", dict_values.len());

                let mut categories_to_bins = AHashMap::new();
                for i in 0..dict_values.len() {
                    if dict_values.is_valid(i) {
                        categories_to_bins.insert(dict_values.value(i).to_string(), i as u8);
                    }
                }

                // dict value at position i always maps to bin i (by construction above)
                let sentinel = categories_to_bins.len() as u8;
                let key_to_bin: Vec<u8> = (0..dict_values.len())
                    .map(|i| if dict_values.is_valid(i) { i as u8 } else { sentinel })
                    .collect();
                let keys = dict.normalized_keys(); // arbitrary values for null rows
                let bins = (0..dict.len())
                    .map(|i| if dict.is_null(i) { sentinel } else { key_to_bin[keys[i]] })
                    .collect();

                Self { bins, kind: BinnerKind::Categorical(categories_to_bins) }
            }
            dt => panic!("unsupported feature type {dt:?}; expected Float64 or Dictionary"),
        }
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self.kind, BinnerKind::Categorical(_))
    }

    pub fn num_bins(&self) -> usize {
        match &self.kind {
            BinnerKind::Numerical(upper_bounds) => upper_bounds.len(),
            BinnerKind::Categorical(cats) => cats.len(),
        }
    }

}

/// Greedy bin boundary search over sorted values.
///
/// Uses an "overshoot guard": before accumulating a value into the current bin,
/// check if doing so would exceed `mean_size`. If so, cut before the value
/// (using the midpoint with the previous distinct value). This naturally isolates
/// high-frequency values into their own bins without a pre-pass. Different logic
/// to lgbm's more complex "is_big" heuristic, but achieves the same goal in practice.
fn greedy_find_bins(sorted_values: &[f64], num_bins: usize, min_data_in_bin: usize) -> Vec<f64> {
    if sorted_values.is_empty() {
        return vec![f64::INFINITY];
    }

    // Compress sorted data into (value, count) pairs. Requires sorted input.
    let distinct: Vec<(f64, usize)> = sorted_values
        .chunk_by(|a, b| a == b)
        .map(|c| (c[0], c.len()))
        .collect();

    let mean_size = (sorted_values.len() as f64 / num_bins as f64).max(min_data_in_bin as f64);
    let mut bounds = Vec::new();
    let mut current_count = 0usize;

    for i in 0..distinct.len() - 1 {
        let (val, count) = distinct[i];

        // Overshoot guard: if adding this value would exceed mean_size, cut before it.
        // Safe to use distinct[i - 1] because current_count > 0 implies i >= 1.
        if current_count > 0 && (current_count + count) as f64 >= mean_size {
            bounds.push(((distinct[i - 1].0 + val) / 2.0).next_up());
            current_count = 0;
            if bounds.len() >= num_bins - 1 {
                break;
            }
        }

        current_count += count;

        // Standard cut: accumulated bin is full, cut after this value.
        if current_count as f64 >= mean_size {
            bounds.push(((val + distinct[i + 1].0) / 2.0).next_up());
            current_count = 0;
            if bounds.len() >= num_bins - 1 {
                break;
            }
        }
    }

    bounds.push(f64::INFINITY);
    bounds
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Float64Array;

    fn make_array(values: &[f64]) -> Float64Array {
        Float64Array::from(values.to_vec())
    }

    #[test]
    fn test_few_distinct_values() {
        // input [1.0, 2.0, 3.0, 1.0, 2.0] → bins [0, 1, 2, 0, 1]
        let arr = make_array(&[1.0, 2.0, 3.0, 1.0, 2.0]);
        let binner = FeatureBinner::from_array(&arr, 255, 1);
        assert_eq!(binner.num_bins(), 3);
        assert_eq!(binner.bins, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn test_max_number_of_bins_limits_bins() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let binner = FeatureBinner::from_array(&make_array(&values), 10, 1);
        assert!(binner.num_bins() <= 10);
    }

    #[test]
    fn test_monotone_bin_assignment() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let binner = FeatureBinner::from_array(&make_array(&values), 32, 1);
        for w in binner.bins.windows(2) {
            assert!(w[0] <= w[1], "bins not monotone: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_null_nan_sentinel() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(f64::NAN)]);
        let binner = FeatureBinner::from_array(&arr, 255, 1);
        assert_eq!(binner.bins[1], binner.num_bins() as u8);
        assert_eq!(binner.bins[2], binner.num_bins() as u8);
    }

    #[test]
    #[should_panic(expected = "num_bins must be at least 1")]
    fn test_invalid_max_number_of_bins() {
        FeatureBinner::from_array(&make_array(&[1.0, 2.0]), 0, 1);
    }

    #[test]
    fn test_dominant_value_gets_own_bin() {
        // [0]*5 + [1]*5 + [2]*2 + [3]*100 + [4]*8, max_bins=3 → {0,1,2} | {3} | {4}
        let values: Vec<f64> = [(0., 5usize), (1., 5), (2., 2), (3., 100), (4., 8)]
            .iter()
            .flat_map(|&(v, n)| std::iter::repeat(v).take(n))
            .collect();
        let binner = FeatureBinner::from_array(&make_array(&values), 3, 1);
        assert_eq!(binner.num_bins(), 3);
        assert_eq!(binner.bins[10], 0);   // value 2.0
        assert_eq!(binner.bins[12], 1);   // value 3.0
        assert_eq!(binner.bins[112], 2);  // value 4.0
    }

    #[test]
    fn test_min_data_in_bin_reduces_bin_count() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr = make_array(&values);
        let strict = FeatureBinner::from_array(&arr, 255, 10);
        let loose = FeatureBinner::from_array(&arr, 255, 1);
        assert!(strict.num_bins() <= loose.num_bins());
    }
}

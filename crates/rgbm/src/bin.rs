//! Feature binning: converts continuous f64 values and categorical strings to bin indices.

use ahash::AHashMap;

use arrow::array::{Array, DictionaryArray, Float64Array, StringArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, UInt32Type};
use arrow::error::ArrowError;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Bin boundaries for a single feature. The last boundary is always +inf.
/// A value `x` maps to bin `i` if `i` is maximal s.t. `x <= upper_bounds[i]`.
/// Nulls and NaNs map to a sentinel bin beyond the last (index `num_bins()`).
pub struct BinMapper {
    pub upper_bounds: Vec<f64>,
}

impl BinMapper {
    /// Build bin boundaries from a Float64Array. Nulls and NaNs are excluded
    /// from boundary construction and map to a sentinel bin beyond the last.
    pub fn from_array(
        values: &Float64Array,
        num_bins: usize,
        min_data_in_bin: usize,
    ) -> Self {
        assert!(num_bins > 0, "num_bins must be at least 1, got {num_bins}");

        let mut valid: Vec<f64> = values.iter().flatten().filter(|x| !x.is_nan()).collect();

        // Subsample using a simple step function if exceeding the LightGBM default
        let max_sample = 200_000;
        if valid.len() > max_sample {
            // TODO: use booster's seed once we have it
            let mut rng = StdRng::seed_from_u64(0);
            
            // Extract the indices into a Vec and sort them
            let mut indices = rand::seq::index::sample(&mut rng, valid.len(), max_sample).into_vec();
            indices.sort_unstable();  // sort to make memory accesses more sequential
            
            // Now memory access is 100% sequential
            valid = indices.into_iter().map(|i| valid[i]).collect();
        }

        valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        Self {
            upper_bounds: Self::greedy_find_bins(&valid, num_bins, min_data_in_bin),
        }
    }

    /// Map a single value to its bin index via binary search.
    pub fn value_to_bin(&self, value: f64) -> u32 {
        self.upper_bounds.partition_point(|&bound| bound < value) as u32
    }

    /// Map a Float64Array to bin indices. Nulls and NaNs get a sentinel index
    /// one past the last bin.
    pub fn array_to_bins(&self, values: &Float64Array) -> Vec<u32> {
        values
            .iter()
            .map(|v| match v {
                Some(x) if !x.is_nan() => self.value_to_bin(x),
                _ => self.sentinel(),
            })
            .collect()
    }

    /// Greedy bin boundary search over sorted values.
    ///
    /// Uses an "overshoot guard": before accumulating a value into the current bin,
    /// check if doing so would exceed `mean_size`. If so, cut before the value
    /// (using the midpoint with the previous distinct value). This naturally isolates
    /// high-frequency values into their own bins without a pre-pass. Different logic
    /// to lgbm's more complex "is_big" heuristic, but achieves the same goal in practice.
    fn greedy_find_bins(
        sorted_values: &[f64],
        num_bins: usize,
        min_data_in_bin: usize,
    ) -> Vec<f64> {
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
}

/// Categorical feature binner: maps category strings to bin indices.
///
/// The bin index for a category equals its position in the Arrow dictionary values array
/// it was built from. Null rows and unknown categories (unseen during training) map to the
/// missing-value sentinel index `num_bins()`, which is handled like NaN in LightGBM — the
/// tree follows a predetermined direction at each split.
pub struct CatMapper {
    /// Maps each category string to its bin index (its position in the training dictionary).
    pub categories_to_bins: AHashMap<String, u32>,
}

impl CatMapper {
    /// Build a CatMapper from a `Dictionary(UInt32, Utf8)` Arrow array.
    ///
    /// `values[k]` is the category string for key `k`. We build the reverse map `category → k`.
    /// Null entries in the dictionary values (unnamed categories) are skipped.
    pub fn from_dictionary(array: &DictionaryArray<UInt32Type>) -> Self {
        let values = array.values().as_any().downcast_ref::<StringArray>().unwrap();
        let mut categories_to_bins = AHashMap::new();
        for i in 0..values.len() {
            if values.is_valid(i) { // skip null dictionary entries (unnamed categories)
                categories_to_bins.insert(values.value(i).to_string(), i as u32);
            }
        }
        Self { categories_to_bins }
    }

    /// Map a single category string to its bin index.
    /// Unknown categories (unseen during training) map to the sentinel index `num_bins()`.
    pub fn value_to_bin(&self, value: &str) -> u32 {
        *self.categories_to_bins
            .get(value)
            .unwrap_or(&self.sentinel())
    }
}

/// Common interface for numeric (`BinMapper`) and categorical (`CatMapper`) feature binners.
pub trait Binner {
    /// Total number of bins. Nulls and unknowns map to sentinel index `num_bins()`.
    fn num_bins(&self) -> usize;

    /// Sentinel bin index for missing/unknown values, equal to `num_bins()`.
    fn sentinel(&self) -> u32 {
        self.num_bins() as u32
    }

    /// Map an Arrow array to bin indices. For `BinMapper`, expects a `Float64Array`;
    /// for `CatMapper`, expects any Dictionary array (casts to Dictionary(UInt32, Utf8)).
    fn array_to_bins(&self, array: &dyn Array) -> Result<Vec<u32>, ArrowError>;
}

impl Binner for BinMapper {
    fn num_bins(&self) -> usize {
        self.upper_bounds.len()
    }

    fn array_to_bins(&self, array: &dyn Array) -> Result<Vec<u32>, ArrowError> {
        let values = array
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| ArrowError::CastError("expected Float64Array".into()))?;
        Ok(self.array_to_bins(values))
    }
}

impl Binner for CatMapper {
    fn num_bins(&self) -> usize {
        self.categories_to_bins.len()
    }

    fn array_to_bins(&self, array: &dyn Array) -> Result<Vec<u32>, ArrowError> {
        let casted = cast(
            array,
            &DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
        )?;
        let dict = casted.as_any().downcast_ref::<DictionaryArray<UInt32Type>>().unwrap();
        let values = dict.values().as_any().downcast_ref::<StringArray>().unwrap();
        let keys = dict.keys();
        let sentinel = self.sentinel();

        // Build a translation table from incoming dictionary keys to bin indices.
        // Only hashes strings K times (number of unique categories), not N times (number of rows).
        let key_to_bin: Vec<u32> = values
            .iter()
            .map(|v| v.and_then(|s| self.categories_to_bins.get(s).copied()).unwrap_or(sentinel))
            .collect();

        Ok(keys
            .iter()
            .map(|k| k.map_or(sentinel, |k| key_to_bin[k as usize]))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_array(values: &[f64]) -> Float64Array {
        Float64Array::from(values.to_vec())
    }

    #[test]
    fn test_few_distinct_values() {
        let arr = make_array(&[1.0, 2.0, 3.0, 1.0, 2.0]);
        let mapper = BinMapper::from_array(&arr, 255, 1);
        assert_eq!(mapper.num_bins(), 3);
        assert_eq!(mapper.value_to_bin(1.0), 0);
        assert_eq!(mapper.value_to_bin(2.0), 1);
        assert_eq!(mapper.value_to_bin(3.0), 2);
    }

    #[test]
    fn test_max_number_of_bins_limits_bins() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mapper = BinMapper::from_array(&make_array(&values), 10, 1);
        assert!(mapper.num_bins() <= 10);
    }

    #[test]
    fn test_monotone_bin_assignment() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let mapper = BinMapper::from_array(&make_array(&values), 32, 1);
        let bins = mapper.array_to_bins(&make_array(&values));
        for w in bins.windows(2) {
            assert!(w[0] <= w[1], "bins not monotone: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_null_nan_sentinel() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(f64::NAN)]);
        let mapper = BinMapper::from_array(&arr, 255, 1);
        let bins = mapper.array_to_bins(&arr);
        assert_eq!(bins[1], mapper.num_bins() as u32);
        assert_eq!(bins[2], mapper.num_bins() as u32);
    }

    #[test]
    #[should_panic(expected = "num_bins must be at least 1")]
    fn test_invalid_max_number_of_bins() {
        BinMapper::from_array(&make_array(&[1.0, 2.0]), 0, 1);
    }

    #[test]
    fn test_dominant_value_gets_own_bin() {
        // [0]*5 + [1]*5 + [2]*2 + [3]*100 + [4]*8, max_bins=3 → {0,1,2} | {3} | {4}
        let values: Vec<f64> = [(0., 5), (1., 5), (2., 2), (3., 100), (4., 8)]
            .iter()
            .flat_map(|&(v, n)| std::iter::repeat(v).take(n))
            .collect();
        let mapper = BinMapper::from_array(&make_array(&values), 3, 1);
        assert_eq!(mapper.num_bins(), 3);
        assert_eq!(mapper.value_to_bin(2.0), 0);
        assert_eq!(mapper.value_to_bin(3.0), 1);
        assert_eq!(mapper.value_to_bin(4.0), 2);
    }

    #[test]
    fn test_min_data_in_bin_reduces_bin_count() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr = make_array(&values);
        let strict = BinMapper::from_array(&arr, 255, 10);
        let loose = BinMapper::from_array(&arr, 255, 1);
        assert!(strict.num_bins() <= loose.num_bins());
    }
}

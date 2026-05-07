// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

//! Feature binning: converts continuous f64/f32 values and categorical strings
//! (Utf8/LargeUtf8/Utf8View dictionary values) to bin indices. Float32 inputs are
//! cast to Float64 and dictionaries with non-Utf8 values are cast to Dict<_, Utf8>
//! on entry.

use ahash::AHashMap;

use arrow::array::{Array, AsArray};
use arrow::compute::cast;
use arrow::datatypes::{DataType, Float64Type};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::seq::IteratorRandom;

// Hardcoded so indices fit into u8.
const MAX_NUM_BINS: usize = 255;

/// Bin mapping rules. Nulls, NaNs, and unknown categories map to bin index -1 = 255.
#[derive(Clone)]
pub enum FeatureBinner {
    // values are upper bounds. For 3 bins (-inf, 0], (0, inf], and missings, upper
    // bounds would be [0.0, inf].
    Numerical(Vec<f64>),
    // Map from category strings to bin indices. Missings are not category strings but
    // map to the sentinel bin index.
    Categorical(AHashMap<String, u8>),
}

impl FeatureBinner {
    pub fn new(array: &dyn Array, max_bin: usize, min_data_in_bin: usize, seed: u64) -> Self {
        assert!(max_bin <= MAX_NUM_BINS, "max_bin {max_bin} exceeds maximum of {MAX_NUM_BINS}");

        // Cast Float32 to Float64 and dictionaries with non-Utf8 values to
        // Dict<_, Utf8>, so the match below only handles those two types.
        // TODO: This brings an unnecessary copy / allocation. One could implement the
        // binning logic for each type directly to avoid this.
        let casted = match array.data_type() {
            DataType::Float32 => Some(cast(array, &DataType::Float64).unwrap()),
            DataType::Dictionary(k, v) if !matches!(v.as_ref(), DataType::Utf8) => Some(
                cast(array, &DataType::Dictionary(k.clone(), Box::new(DataType::Utf8))).unwrap(),
            ),
            _ => None,
        };
        let array = casted.as_deref().unwrap_or(array);

        match array.data_type() {
            DataType::Float64 => {
                let values = array.as_primitive::<Float64Type>();

                // For very large datasets, use a subsample of the dataset to determine
                // bin boundaries.
                const MAX_SAMPLE: usize = 200_000;  // same as LGBM
                let mut rng = StdRng::seed_from_u64(seed);
                let mut valid = values.iter().flatten().filter(|x| !x.is_nan())
                    .choose_multiple(&mut rng, MAX_SAMPLE);
                valid.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                FeatureBinner::Numerical(greedy_find_bins(&valid, max_bin, min_data_in_bin))
            }
            DataType::Dictionary(_, _) => {
                let dict = array.as_any_dictionary();
                let dict_values = dict.values().as_string::<i32>();

                let mut categories = AHashMap::new();
                let mut bin_idx: u8 = 0;
                for i in 0..dict_values.len() {
                    // missings are implicit and not part of the dict_values.
                    // They get skipped.
                    if dict_values.is_valid(i) {
                        categories.insert(dict_values.value(i).to_string(), bin_idx);
                        bin_idx += 1;
                    }
                }

                // todo: implement multi-category splits, either by grouping least
                // frequent categories into "other", or some other magic.
                // Putting the assert! after the loop makes it easier to handle the edge
                // cases: (i) 256 categories, 1 of them missings (allowed) and (ii)
                // 256 categories, all valid (not allowed).
                assert!(
                    categories.len() < max_bin,
                    "categorical feature has {} valid categories, max allowed is {}; reduce the number of categories",
                    categories.len(),
                    max_bin - 1
                );
                FeatureBinner::Categorical(categories)
            }
            dt => panic!("unsupported feature type {dt:?}; expected Float64, Float32, or Dictionary with Utf8/LargeUtf8/Utf8View values"),
        }
    }

    /// Apply the binner an Arrow array, producing one bin index per row.
    pub fn apply(&self, array: &dyn Array) -> Vec<u8> {
        // Same casts as in new(): Float32 -> Float64 and Dict<_, non-Utf8> ->
        // Dict<_, Utf8>. Float64 / Dict<_, Utf8> inputs skip this.
        // todo: avoid the per-call Vec<usize> allocation in normalized_keys() (in the
        // categorical arm) by matching on the concrete key type and indexing the typed
        // buffer directly.
        let casted = match array.data_type() {
            DataType::Float32 => Some(cast(array, &DataType::Float64).unwrap()),
            DataType::Dictionary(k, v) if !matches!(v.as_ref(), DataType::Utf8) => Some(
                cast(array, &DataType::Dictionary(k.clone(), Box::new(DataType::Utf8))).unwrap(),
            ),
            _ => None,
        };
        let array = casted.as_deref().unwrap_or(array);

        match self {
            FeatureBinner::Numerical(upper_bounds) => {
                let values = array.as_primitive::<Float64Type>();
                let raw_values = values.values();

                // unknown bins map past the last index. Equal to self::num_bins() - 1.
                let sentinel = upper_bounds.len() as u8;
                let mut binned_values = Vec::with_capacity(raw_values.len());

                // Check for nulls outside the main loop to avoid branches. Nulls are
                // stored in a separate bitmap. NaNs are specific floats. We treat both
                // the same. The loop without nulls is much faster.
                if let Some(nulls) = values.nulls() {
                    for (i, &x) in raw_values.iter().enumerate() {
                        binned_values.push(if nulls.is_null(i) || x.is_nan() { sentinel } else { upper_bounds.partition_point(|&b| b < x) as u8 });
                    }
                } else {
                    for &x in raw_values {
                        binned_values.push(if x.is_nan() { sentinel } else { upper_bounds.partition_point(|&b| b < x) as u8 });
                    }
                }
                binned_values
            }
            FeatureBinner::Categorical(categories) => {
                let dict = array.as_any_dictionary();
                let dict_values = dict.values().as_string::<i32>();
                // unknown categories map to last bin index. This needs not be 255 if
                // there are fewer than 255 categories, allowing us to use tighter bin
                // packing in the future (todo).
                let sentinel = categories.len() as u8;

                // Build a mapping from dictionary key to bin index. Importantly,
                // calling apply(array) on an array with different categories as the one
                // used in new() still works.
                let key_to_bin: Vec<u8> = dict_values
                    .iter()
                    .map(|opt_val| {
                        opt_val
                            .and_then(|val| categories.get(val).copied())
                            .unwrap_or(sentinel)
                    })
                    .collect();

                let keys = dict.normalized_keys();
                let mut binned_values = Vec::with_capacity(dict.len());
                
                // Same as for numericals: Check for nulls outside the main loop.
                if let Some(nulls) = dict.nulls() {
                    for i in 0..dict.len() {
                        binned_values.push(if nulls.is_null(i) { 
                            sentinel 
                        } else { 
                            key_to_bin[keys[i] as usize] 
                        });
                    }
                } else {
                    for key in keys {
                        binned_values.push(key_to_bin[key as usize]);
                    }
                }
                binned_values
            }
        }
    }

    // Number of bins, including the sentinel bin.
    pub fn num_bins(&self) -> usize {
        match self {
            // For bins (-inf, 0], (0, inf], and missings -> upper_bounds = [0.0, inf],
            // num_bins = 3.
            FeatureBinner::Numerical(upper_bounds) => upper_bounds.len() + 1,
            // +1 as missings and unknown categories are implicit.
            FeatureBinner::Categorical(cats) => cats.len() + 1,
        }
    }

    pub fn is_categorical(&self) -> bool {
        matches!(self, FeatureBinner::Categorical(_))
    }
}

/// Greedy bin boundary search over sorted values.
///
/// Uses an "overshoot guard": before accumulating a value into the current bin,
/// check if doing so would exceed `mean_size`. If so, cut before the value
/// (using the midpoint with the previous distinct value). This naturally isolates
/// high-frequency values into their own bins without a pre-pass. Different logic
/// to lgbm's more complex "is_big" heuristic, but achieves the same goal in practice.
fn greedy_find_bins(sorted_values: &[f64], max_bin: usize, min_data_in_bin: usize) -> Vec<f64> {
    if sorted_values.is_empty() {
        return vec![f64::INFINITY];
    }

    // Compress sorted data into (value, count) pairs. Requires sorted input.
    let value_counts: Vec<(f64, usize)> = sorted_values
        .chunk_by(|a, b| a == b)
        .map(|c| (c[0], c.len()))
        .collect();

    let mean_size = (sorted_values.len() as f64 / max_bin as f64).max(min_data_in_bin as f64);
    let mut bounds = Vec::new();
    let mut current_count: usize = 0;

    for i in 0..value_counts.len() - 1 {
        let (value, count) = value_counts[i];

        // Overshoot guard: if adding this value to current bin would exceed mean_size,
        // cut before it.
        // Safe to use value_counts[i - 1] because current_count > 0 implies i >= 1.
        if current_count > 0 && (current_count + count) as f64 >= mean_size {
            bounds.push(((value_counts[i - 1].0 + value) / 2.0).next_up());
            current_count = 0;
            if bounds.len() >= max_bin - 1 {
                break;
            }
        }

        current_count += count;

        // Standard cut: accumulated bin is full, cut after this value.
        if current_count as f64 >= mean_size {
            bounds.push(((value + value_counts[i + 1].0) / 2.0).next_up());
            current_count = 0;
            if bounds.len() >= max_bin - 1 {
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
    use arrow::array::{
        DictionaryArray, Float32Array, Float64Array, LargeStringArray, StringArray,
        StringViewArray, UInt32Array,
    };
    use arrow::datatypes::UInt32Type;
    use std::sync::Arc;

    fn make_array(values: &[f64]) -> Float64Array {
        Float64Array::from(values.to_vec())
    }

    #[test]
    fn test_few_distinct_values() {
        // input [1.0, 2.0, 3.0, 1.0, 2.0] → bins [0, 1, 2, 0, 1], sentinel at 3
        let arr = make_array(&[1.0, 2.0, 3.0, 1.0, 2.0]);
        let binner = FeatureBinner::new(&arr, 255, 1, 0);
        assert_eq!(binner.apply(&arr), vec![0, 1, 2, 0, 1]);
        assert_eq!(binner.num_bins(), 4);
    }

    #[test]
    fn test_monotone_bin_assignment() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let arr = make_array(&values);
        let binner = FeatureBinner::new(&arr, 255, 1, 0);
        let bins = binner.apply(&arr);
        for w in bins.windows(2) {
            assert!(w[0] <= w[1], "bins not monotone: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_null_nan_sentinel() {
        let arr = Float64Array::from(vec![Some(1.0), None, Some(f64::NAN)]);
        let binner = FeatureBinner::new(&arr, 255, 1, 0);
        let bins = binner.apply(&arr);
        // null and NaN map to the same sentinel bin
        assert_eq!(bins[1], bins[2]);
        // sentinel is below num_bins() (which counts sentinel as a bin)
        assert!((bins[1] as usize) < binner.num_bins());
        // valid value maps to a different bin
        assert_ne!(bins[0], bins[1]);

    }

    #[test]
    fn test_min_data_in_bin_reduces_bin_count() {
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let arr = make_array(&values);
        let strict = FeatureBinner::new(&arr, 255, 10, 0);
        let loose = FeatureBinner::new(&arr, 255, 1, 0);
        assert!(strict.num_bins() <= loose.num_bins());
    }

    #[test]
    fn test_float32_via_cast() {
        // f32 input: must give the same bins as the equivalent f64 input.
        let f32_arr = Float32Array::from(vec![Some(1.0f32), Some(2.0), None, Some(f32::NAN), Some(3.0)]);
        let f64_arr = Float64Array::from(vec![Some(1.0f64), Some(2.0), None, Some(f64::NAN), Some(3.0)]);
        let f32_binner = FeatureBinner::new(&f32_arr, 255, 1, 0);
        let f64_binner = FeatureBinner::new(&f64_arr, 255, 1, 0);
        assert_eq!(f32_binner.apply(&f32_arr), f64_binner.apply(&f64_arr));
        // null and NaN map to sentinel.
        let bins = f32_binner.apply(&f32_arr);
        assert_eq!(bins[2], bins[3]);
        assert!((bins[2] as usize) < f32_binner.num_bins());
    }

    fn make_utf8_dict(keys: &[u32], values: &[&str]) -> DictionaryArray<UInt32Type> {
        let keys = UInt32Array::from(keys.to_vec());
        let values = Arc::new(StringArray::from(values.to_vec()));
        DictionaryArray::<UInt32Type>::try_new(keys, values).unwrap()
    }

    #[test]
    fn test_large_utf8_dict_via_cast() {
        // Polars-style Dict<UInt32, LargeUtf8>: must bin the same as Dict<UInt32, Utf8>.
        let keys = UInt32Array::from(vec![0u32, 1, 2, 0, 1]);
        let large = Arc::new(LargeStringArray::from(vec!["a", "b", "c"]));
        let large_dict = DictionaryArray::<UInt32Type>::try_new(keys.clone(), large).unwrap();
        let utf8_dict = make_utf8_dict(&[0, 1, 2, 0, 1], &["a", "b", "c"]);

        let large_binner = FeatureBinner::new(&large_dict, 255, 1, 0);
        let utf8_binner = FeatureBinner::new(&utf8_dict, 255, 1, 0);
        assert_eq!(large_binner.apply(&large_dict), utf8_binner.apply(&utf8_dict));
        assert_eq!(large_binner.num_bins(), 4); // 3 categories + sentinel
    }

    #[test]
    fn test_utf8_view_dict_via_cast() {
        // Newer-polars-style Dict<UInt32, Utf8View>: must bin the same as Dict<UInt32, Utf8>.
        let keys = UInt32Array::from(vec![0u32, 1, 2, 0, 1]);
        let view = Arc::new(StringViewArray::from(vec!["a", "b", "c"]));
        let view_dict = DictionaryArray::<UInt32Type>::try_new(keys.clone(), view).unwrap();
        let utf8_dict = make_utf8_dict(&[0, 1, 2, 0, 1], &["a", "b", "c"]);

        let view_binner = FeatureBinner::new(&view_dict, 255, 1, 0);
        let utf8_binner = FeatureBinner::new(&utf8_dict, 255, 1, 0);
        assert_eq!(view_binner.apply(&view_dict), utf8_binner.apply(&utf8_dict));
        assert_eq!(view_binner.num_bins(), 4);
    }
}

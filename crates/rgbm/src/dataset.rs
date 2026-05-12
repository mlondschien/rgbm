// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use arrow::array::{Float64Array, RecordBatch};
use rayon::prelude::*;

use crate::bin::FeatureBinner;
use crate::parameters::DatasetParameters;
use crate::utils::build_thread_pool;

/// FeatureBundle stores the features' bin indices.
/// Bin values are u8. We pack up to 8 of these into a single u64 (1 byte each).
/// This improves memory bandwidth in histogram building.
pub struct FeatureBundle {
    pub packed_bins: Vec<u64>,
    pub num_bins: Vec<usize>,  // including the sentinel bin.
    pub is_categorical: Vec<bool>,
    pub count: usize, // number of features in the bundle.
}

impl FeatureBundle {
    fn pack(binners: &[FeatureBinner], bins: &[Vec<u8>], num_rows: usize) -> Self {
        let mut packed_bins = vec![0u64; num_rows];
        for (slot, col) in bins.iter().enumerate() {
            let shift = slot * 8;
            for row in 0..num_rows {
                packed_bins[row] |= (col[row] as u64) << shift;
            }
        }
        let num_bins: Vec<usize> = binners.iter().map(|b| b.num_bins()).collect();
        let is_categorical: Vec<bool> = binners.iter().map(|b| b.is_categorical()).collect();
        Self { packed_bins, num_bins, is_categorical, count: binners.len() }
    }
}

/// Training dataset: stores binned features, labels, and optional weights / offsets.
pub struct Dataset {
    pub feature_binners: Vec<FeatureBinner>,
    pub feature_bundles: Vec<FeatureBundle>,
    pub feature_names: Vec<String>,
    pub labels: Float64Array,
    pub weights: Option<Float64Array>,
    pub offsets: Option<Float64Array>,
    pub num_rows: usize,
}

impl Dataset {
    /// Build a `Dataset` from an Arrow `RecordBatch` of features plus separate label/weight/offsets arrays.
    pub fn from_arrow(
        features: &RecordBatch,
        labels: &Float64Array,
        weights: Option<&Float64Array>,
        offsets: Option<&Float64Array>,
        params: &DatasetParameters,
    ) -> Self {
        let num_features = features.num_columns();
        let num_rows = features.num_rows();

        let feature_names: Vec<String> = features.schema().fields().iter()
            .map(|f| f.name().clone())
            .collect();

        let pool = build_thread_pool(params.n_jobs);

        let (feature_binners, all_bins): (Vec<_>, Vec<_>) = match &pool {
            Some(pool) => pool.install(|| {
                features.columns().par_iter().map(|array| {
                    let binner = FeatureBinner::new(array.as_ref(), params.max_bin, params.min_data_in_bin, params.seed);
                    let bins = binner.apply(array.as_ref());
                    (binner, bins)
                }).unzip()
            }),
            None => features.columns().iter().map(|array| {
                let binner = FeatureBinner::new(array.as_ref(), params.max_bin, params.min_data_in_bin, params.seed);
                let bins = binner.apply(array.as_ref());
                (binner, bins)
            }).unzip(),
        };

        // Pack features into bundles of 8
        let num_chunks = num_features.div_ceil(8);
        let feature_bundles: Vec<FeatureBundle> = match &pool {
            Some(pool) => pool.install(|| {
                (0..num_chunks).into_par_iter().map(|chunk_idx| {
                    let start = chunk_idx * 8;
                    let end = (start + 8).min(num_features);
                    FeatureBundle::pack(&feature_binners[start..end], &all_bins[start..end], num_rows)
                }).collect()
            }),
            None => (0..num_chunks).map(|chunk_idx| {
                let start = chunk_idx * 8;
                let end = (start + 8).min(num_features);
                FeatureBundle::pack(&feature_binners[start..end], &all_bins[start..end], num_rows)
            }).collect(),
        };

        Self {
            feature_binners,
            feature_bundles,
            feature_names,
            labels: labels.clone(),
            weights: weights.cloned(),
            offsets: offsets.cloned(),
            num_rows,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{DictionaryArray, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
    use std::sync::Arc;

    fn make_features() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("x", DataType::Float64, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]))],
        )
        .unwrap()
    }

    #[test]
    fn test_basic_dataset() {
        let labels = Float64Array::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
        let params = DatasetParameters { min_data_in_bin: 1, ..DatasetParameters::default() };
        let ds = Dataset::from_arrow(&make_features(), &labels, None, None, &params);
        assert_eq!(ds.num_rows, 5);
        assert_eq!(ds.feature_binners.len(), 1);
        assert_eq!(ds.feature_names, vec!["x"]);
        assert_eq!(ds.labels, Float64Array::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]));
    }

    #[test]
    fn test_categorical_feature() {
        let keys = UInt32Array::from(vec![0u32, 1, 0, 2, 1]);
        let values = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let dict = DictionaryArray::<UInt32Type>::try_new(keys, values).unwrap();

        let schema = Schema::new(vec![Field::new(
            "cat",
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            false,
        )]);
        let features = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(dict)]).unwrap();
        let labels = Float64Array::from(vec![0.0, 1.0, 0.0, 1.0, 0.0]);

        let params = DatasetParameters { min_data_in_bin: 1, ..DatasetParameters::default() };
        let ds = Dataset::from_arrow(&features, &labels, None, None, &params);
        assert_eq!(ds.feature_binners.len(), 1);
        assert!(matches!(ds.feature_binners[0], FeatureBinner::Categorical(_)));
        assert_eq!(ds.feature_binners[0].num_bins(), 4); // 3 categories + sentinel
    }
}

use arrow::array::{Float64Array, RecordBatch};

use crate::bin::FeatureBinner;

/// FeatureBundle stores the features' bin indices.
/// Bin values are u8. Like LGBM, we pack up to 4 of these into a single u32 (1 byte).
/// This improves memory bandwidth in histogram builing, the part of the algorithm where
/// >90% of time is spent.
pub struct FeatureBundle {
    pub packed_bins: Vec<u32>,
    pub feature_indices: Vec<usize>,
    pub num_bins: Vec<usize>,
    pub is_categorical: Vec<bool>,
    pub count: usize, // number of features in the bundle
}

impl FeatureBundle {
    fn pack(binners: &[&FeatureBinner], bins: &[Vec<u8>], feature_indices: Vec<usize>, num_rows: usize) -> Self {
        let count = binners.len();
        let mut packed_bins = vec![0u32; num_rows];
        for (slot, col) in bins.iter().enumerate() {
            let shift = slot * 8;
            for row in 0..num_rows {
                packed_bins[row] |= (col[row] as u32) << shift;
            }
        }
        let num_bins: Vec<usize> = binners.iter().map(|b| b.num_bins()).collect();
        let is_categorical: Vec<bool> = binners.iter().map(|b| b.is_categorical()).collect();
        Self { packed_bins, feature_indices, num_bins, is_categorical, count }
    }
}

/// Training dataset: stores binned features, labels, and optional weights.
pub struct Dataset {
    pub feature_binners: Vec<FeatureBinner>,
    pub feature_bundles: Vec<FeatureBundle>,
    pub feature_names: Vec<String>,
    pub labels: Float64Array,
    pub weights: Option<Float64Array>,
    pub num_rows: usize,
    pub num_features: usize,
}

impl Dataset {
    /// Build a `Dataset` from an Arrow `RecordBatch` of features plus separate label/weight arrays.
    pub fn from_arrow(
        features: &RecordBatch,
        labels: &Float64Array,
        weights: Option<&Float64Array>,
        max_bin: usize,
        min_data_in_bin: usize,
    ) -> Self {
        let num_features = features.num_columns();
        let mut feature_binners = Vec::with_capacity(num_features);
        let mut feature_names = Vec::with_capacity(num_features);
        let mut all_bins: Vec<Vec<u8>> = Vec::with_capacity(num_features);

        for (field, array) in features.schema().fields().iter().zip(features.columns()) {
            feature_names.push(field.name().clone());
            let binner = FeatureBinner::new(array.as_ref(), max_bin, min_data_in_bin);
            all_bins.push(binner.apply(array.as_ref()));
            feature_binners.push(binner);
        }

        let num_rows = features.num_rows();

        let binner_refs: Vec<&FeatureBinner> = feature_binners.iter().collect();
        let feature_bundles: Vec<FeatureBundle> = binner_refs.chunks(4)
            .zip(all_bins.chunks(4))
            .enumerate()
            .map(|(chunk_idx, (chunk_binners, chunk_bins))| {
                let start = chunk_idx * 4;
                let feature_indices = (start..start + chunk_binners.len()).collect();
                FeatureBundle::pack(chunk_binners, chunk_bins, feature_indices, num_rows)
            })
            .collect();

        Self {
            feature_binners,
            feature_bundles,
            feature_names,
            labels: labels.clone(),
            weights: weights.cloned(),
            num_rows,
            num_features,
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
        let ds = Dataset::from_arrow(&make_features(), &labels, None, 255, 1);
        assert_eq!(ds.num_rows, 5);
        assert_eq!(ds.num_features, 1);
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

        let ds = Dataset::from_arrow(&features, &labels, None, 255, 1);
        assert_eq!(ds.num_features, 1);
        assert!(matches!(ds.feature_binners[0], FeatureBinner::Categorical(_)));
        assert_eq!(ds.feature_binners[0].num_bins(), 4); // 3 categories + sentinel
    }
}

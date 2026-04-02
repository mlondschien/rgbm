use arrow::array::{Float64Array, RecordBatch};

use crate::bin::FeatureBinner;

/// Training dataset: stores binned features, labels, and optional weights.
pub struct Dataset {
    pub feature_binners: Vec<FeatureBinner>,
    pub feature_names: Vec<String>,
    pub labels: Float64Array,
    pub weights: Option<Float64Array>,
    pub num_rows: usize,
    pub num_features: usize,
    pub max_bins: usize,
}

impl Dataset {
    /// Build a `Dataset` from an Arrow `RecordBatch` of features plus separate label/weight arrays.
    pub fn from_arrow(
        features: &RecordBatch,
        labels: &Float64Array,
        weights: Option<&Float64Array>,
        num_bins: usize,
        min_data_in_bin: usize,
    ) -> Self {
        let num_features = features.num_columns();
        let mut feature_binners = Vec::with_capacity(num_features);
        let mut feature_names = Vec::with_capacity(num_features);

        for (field, array) in features.schema().fields().iter().zip(features.columns()) {
            feature_names.push(field.name().clone());
            feature_binners.push(FeatureBinner::from_array(array.as_ref(), num_bins, min_data_in_bin));
        }

        let max_bins = feature_binners.iter().map(|b| b.num_bins()).max().unwrap_or(num_bins);

        Self {
            feature_binners,
            feature_names,
            labels: labels.clone(),
            weights: weights.cloned(),
            num_rows: features.num_rows(),
            num_features,
            max_bins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{DictionaryArray, StringArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
    use std::sync::Arc;
    use crate::bin::BinnerKind;

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
        assert!(matches!(ds.feature_binners[0].kind, BinnerKind::Categorical(_)));
        assert_eq!(ds.feature_binners[0].num_bins(), 3);
    }
}

use arrow::array::{Array, AsArray, Float64Array, RecordBatch};
use arrow::datatypes::DataType;
use arrow::error::ArrowError;

use crate::bin::{Binner, BinMapper, CatMapper};

/// Wraps either a numeric (`BinMapper`) or categorical (`CatMapper`) binner for a single feature.
pub enum FeatureBinner {
    Numeric(BinMapper),
    Categorical(CatMapper),
}

impl Binner for FeatureBinner {
    fn num_bins(&self) -> usize {
        match self {
            Self::Numeric(b) => b.num_bins(),
            Self::Categorical(b) => b.num_bins(),
        }
    }

    fn array_to_bins(&self, array: &dyn Array) -> Result<Vec<u16>, ArrowError> {
        match self {
            Self::Numeric(b) => b.array_to_bins(array),
            Self::Categorical(b) => b.array_to_bins(array),
        }
    }
}

/// Training dataset: stores binned features, labels, and optional weights.
pub struct Dataset {
    /// `binned_features[j][i]` is the bin index for row `i`, feature `j`.
    pub binned_features: Vec<Vec<u16>>,
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
        let mut binned_features = Vec::with_capacity(num_features);
        let mut feature_names = Vec::with_capacity(num_features);

        for (field, array) in features.schema().fields().iter().zip(features.columns()) {
            let name = field.name();

            let binner = if array.data_type() == &DataType::Float64 {
                FeatureBinner::Numeric(BinMapper::from_array(array.as_primitive(), num_bins, min_data_in_bin))
            } else if let DataType::Dictionary(_, _) = array.data_type() {
                FeatureBinner::Categorical(CatMapper::from_dictionary(array.as_ref()))
            } else {
                panic!("column '{name}' has unsupported type {:?}; expected Float64 or Dictionary", array.data_type());
            };

            let bins = binner.array_to_bins(array.as_ref()).unwrap();
            feature_names.push(name.clone());
            feature_binners.push(binner);
            binned_features.push(bins);
        }

        let max_bins = feature_binners.iter().map(|b| b.num_bins()).max().unwrap_or(num_bins);

        Self {
            binned_features,
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
    use arrow::datatypes::{Field, Schema, UInt32Type};
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
        assert_eq!(ds.feature_binners[0].num_bins(), 3);
    }
}

use std::sync::Arc;
use arrow::array::{DictionaryArray, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, UInt32Type};
use arrow::record_batch::RecordBatch;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

use rgbm::booster::Booster;
use rgbm::dataset::Dataset;
use rgbm::objective::SquaredLoss;
use rgbm::parameters::Parameters;

const N: usize = 1_000_000;
const P_NUM: usize = 40;
const P_CAT: usize = 10;
const N_CATEGORIES: u32 = 20;

fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    let labels: Vec<f64> = (0..N).map(|_| rng.r#gen::<f64>()).collect();

    let mut fields: Vec<Field> = (0..P_NUM)
        .map(|i| Field::new(format!("f{i}"), DataType::Float64, false))
        .collect();
    let cat_type = DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8));
    for i in 0..P_CAT {
        fields.push(Field::new(format!("c{i}"), cat_type.clone(), false));
    }

    let mut columns: Vec<Arc<dyn arrow::array::Array>> = (0..P_NUM)
        .map(|_| {
            let col: Vec<f64> = (0..N).map(|_| rng.r#gen::<f64>()).collect();
            Arc::new(Float64Array::from(col)) as _
        })
        .collect();

    let cat_values = Arc::new(StringArray::from(
        (0..N_CATEGORIES).map(|i| format!("cat_{i}")).collect::<Vec<_>>()
    ));
    for _ in 0..P_CAT {
        let keys = UInt32Array::from((0..N).map(|_| rng.r#gen::<u32>() % N_CATEGORIES).collect::<Vec<_>>());
        let dict = DictionaryArray::<UInt32Type>::try_new(keys, cat_values.clone()).unwrap();
        columns.push(Arc::new(dict) as _);
    }

    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns).unwrap();
    let labels_arr = Float64Array::from(labels);

    let dataset = Dataset::from_arrow(&batch, &labels_arr, None, 255, 20);

    let params = Parameters {
        num_iterations: 100,
        learning_rate: 0.1,
        max_depth: 6,
        max_leaves: 31,
lambda_l2: 1.0,
        ..Parameters::default()
    };

    let mut booster = Booster::new(params, Box::new(SquaredLoss));
    booster.fit(&dataset);

    // prevent dead code elimination
    let pred = booster.predict(&batch);
    println!("first prediction: {:.4}", pred.value(0));
}

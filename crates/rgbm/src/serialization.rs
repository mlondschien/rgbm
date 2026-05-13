// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

//! LightGBM `model.txt` (v4) compatible serialization.

use crate::bin::FeatureBinner;
use crate::booster::Booster;
use crate::histogram::Threshold;
use crate::tree::{Node, Tree};
use std::fmt::Write;

impl Booster {
    pub fn model_to_string(&self) -> String {
        // LightGBM has no `base_score` field, instead it prepends a 1-leaf tree with
        // leaf-value = base_score. We do the same.
        let mut tree_texts: Vec<String> = Vec::with_capacity(self.trees.len() + 1);
        tree_texts.push(format!(
            "\
Tree=0
num_leaves=1
num_cat=0
split_feature=
split_gain=
threshold=
decision_type=
left_child=
right_child=
leaf_value={value}
leaf_weight=0
leaf_count=0
internal_value=
internal_weight=
internal_count=
is_linear=0
shrinkage=1


",
            value = self.base_score
        ));

        for (i, tree) in self.trees.iter().enumerate() {
            let mut t = format!("Tree={}\n", i + 1);
            tree.write_to_string(&mut t, &self.feature_binners);
            t.push_str("\n\n");
            tree_texts.push(t);
        }

        let feature_infos: Vec<String> = self
            .feature_binners
            .iter()
            .map(|b| match b {
                FeatureBinner::Numerical(_) => "[-inf:inf]".to_string(),
                FeatureBinner::Categorical(map) => std::iter::once("-1".to_string())
                    .chain((0..map.len()).map(|i| i.to_string()))
                    .collect::<Vec<_>>()
                    .join(":"),
            })
            .collect();

        let objective_name = self.objective.lgbm_name();
        let max_feature_indices = self.feature_binners.len() - 1;
        let feature_names = self.feature_names.join(" ");
        let feature_infos = feature_infos.join(" ");
        let tree_sizes = tree_texts
            .iter()
            .map(|t| t.len().to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let mut s = format!(
            "\
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx={max_feature_indices}
objective={objective_name}
feature_names={feature_names}
feature_infos={feature_infos}
tree_sizes={tree_sizes}

",
        );
        for t in &tree_texts {
            s.push_str(t);
        }
        write!(
            s,
            "\
end of trees

parameters:
[boosting: gbdt]
[objective: {objective_name}]
[num_class: 1]
[num_tree_per_iteration: 1]
end of parameters

pandas_categorical:null
"
        )
        .unwrap();
        s
    }
}

impl Tree {
    pub(crate) fn write_to_string(&self, s: &mut String, binners: &[FeatureBinner]) {
        // LightGBM's child indexing: i for internal node i, -(j+1) for leaf j.
        let mut node_idx = vec![0i32; self.nodes.len()];
        let mut leaf_values = Vec::new();
        let mut n_internal = 0i32;
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::Internal { .. } => {
                    node_idx[i] = n_internal;
                    n_internal += 1;
                }
                Node::Leaf { value } => {
                    node_idx[i] = -(leaf_values.len() as i32 + 1);
                    leaf_values.push(*value);
                }
            }
        }

        let mut split_feature = Vec::new();
        let mut threshold = Vec::new();
        let mut left_child = Vec::new();
        let mut right_child = Vec::new();
        let mut decision_type: Vec<u8> = Vec::new();
        let mut cat_thresholds: Vec<u32> = Vec::new();
        let mut cat_boundaries: Vec<usize> = vec![0];

        for node in &self.nodes {
            let Node::Internal {
                left_child: lc,
                right_child: rc,
                split_feature: sf,
                threshold: t,
            } = node
            else {
                continue;
            };
            split_feature.push(*sf);
            left_child.push(node_idx[*lc as usize]);
            right_child.push(node_idx[*rc as usize]);

            // bit 0: categorical, bit 1: default_left, bits 2-3: missing_type (NaN=2)
            let mut d = 8u8;
            match t {
                Threshold::Numeric {
                    bin,
                    missing_goes_left,
                } => {
                    if *missing_goes_left {
                        d |= 2;
                    }
                    let FeatureBinner::Numerical(upper_bounds) = &binners[*sf as usize] else {
                        unreachable!()
                    };
                    threshold.push(upper_bounds[*bin as usize]);
                }
                Threshold::Categorical(cats) => {
                    d |= 1;
                    // cats[sentinel_bin] encodes missing routing; lgbm conveys that via
                    // the decision_type bit, so we skip that slot when packing.
                    let sentinel_bin = cats.len() - 1;
                    if cats[sentinel_bin] {
                        d |= 2;
                    }
                    threshold.push((cat_boundaries.len() - 1) as f64);
                    // Pack directly into the lgbm cat_threshold format (8 u32 words = 256 bits).
                    let start = cat_thresholds.len();
                    cat_thresholds.resize(start + 8, 0);
                    for (i, &left) in cats.iter().enumerate() {
                        if left && i != sentinel_bin {
                            cat_thresholds[start + i / 32] |= 1u32 << (i % 32);
                        }
                    }
                    cat_boundaries.push(cat_thresholds.len());
                }
            }
            decision_type.push(d);
        }

        fn join<T: std::fmt::Display>(v: &[T]) -> String {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        }
        let n_internal = split_feature.len();
        let n_leaf = leaf_values.len();
        write!(
            s,
            "\
num_leaves={n_leaf}
num_cat={n_cat}
split_feature={split_feature}
split_gain={zeros_internal}
threshold={threshold}
decision_type={decision_type}
left_child={left_child}
right_child={right_child}
leaf_value={leaf_value}
leaf_weight={zeros_leaf}
leaf_count={zeros_leaf}
internal_value={zeros_internal}
internal_weight={zeros_internal}
internal_count={zeros_internal}
",
            n_cat = cat_boundaries.len() - 1,
            split_feature = join(&split_feature),
            threshold = join(&threshold),
            decision_type = join(&decision_type),
            left_child = join(&left_child),
            right_child = join(&right_child),
            leaf_value = join(&leaf_values),
            zeros_internal = join(&vec![0u8; n_internal]),
            zeros_leaf = join(&vec![0u8; n_leaf]),
        )
        .unwrap();
        if !cat_thresholds.is_empty() {
            writeln!(s, "cat_boundaries={}", join(&cat_boundaries)).unwrap();
            writeln!(s, "cat_threshold={}", join(&cat_thresholds)).unwrap();
        }
        s.push_str("is_linear=0\nshrinkage=1\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::objective::Gaussian;
    use crate::parameters::{BoosterParameters, DatasetParameters};
    use arrow::array::{Float64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_model_to_string_structure() {
        // Fit a tiny booster and check structural invariants of the serialized model.
        let n = 50;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(x))]).unwrap();
        let labels = Float64Array::from(y);
        let params = DatasetParameters {
            min_data_in_bin: 1,
            ..DatasetParameters::default()
        };
        let dataset = Dataset::from_arrow(&batch, &labels, None, None, &params);

        let num_iterations = 5;
        let mut booster = Booster::new(
            BoosterParameters {
                num_iterations,
                min_sum_hessian_in_leaf: 0.0,
                ..BoosterParameters::default()
            },
            Box::new(Gaussian),
        );
        booster.fit(&dataset);

        let s = booster.model_to_string();

        // Top-level scaffolding.
        assert!(s.starts_with("tree\nversion=v4\n"));
        assert!(s.contains("end of trees\n"));
        assert!(s.contains("end of parameters\n"));
        assert!(s.ends_with("pandas_categorical:null\n"));
        assert!(s.contains("objective=regression"));

        // One "Tree=N" header for the base-score tree + each fit tree.
        let tree_headers =
            s.matches("\nTree=").count() + s.starts_with("Tree=").then_some(1).unwrap_or(0);
        // model_to_string puts Tree=N at start of each tree block; first one after the
        // header is "Tree=0" (base score), then "Tree=1".."Tree=num_iterations".
        assert_eq!(tree_headers, num_iterations + 1);

        // Each tree section has exactly one num_leaves= line.
        assert_eq!(s.matches("num_leaves=").count(), num_iterations + 1);
    }
}

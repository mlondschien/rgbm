// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

//! LightGBM `model.txt` (v4) compatible serialization.

use std::fmt::Write;
use crate::bin::FeatureBinner;
use crate::booster::Booster;
use crate::tree::{Node, Threshold, Tree};

impl Booster {
    pub fn model_to_string(&self) -> String {
        // LightGBM has no `base_score` field, so we prepend a 1-leaf tree with
        // value=base_score. Then pred = sum(leaf values) matches our
        // pred = base_score + sum(tree deltas).
        let mut tree_texts: Vec<String> = Vec::with_capacity(self.trees.len() + 1);
        let mut t0 = String::from("Tree=0\n");
        write_constant_tree(&mut t0, self.base_score);
        t0.push_str("\n\n");
        tree_texts.push(t0);
        for (i, tree) in self.trees.iter().enumerate() {
            let mut t = format!("Tree={}\n", i + 1);
            tree.write_to_string(&mut t);
            t.push_str("\n\n");
            tree_texts.push(t);
        }

        let feature_infos: Vec<String> = self.feature_binners.iter().map(|b| match b {
            FeatureBinner::Numerical(upper) => {
                let lo = upper.first().copied().unwrap_or(0.0);
                let hi = if upper.len() >= 2 { upper[upper.len() - 2] } else { lo };
                format!("[{lo}:{hi}]")
            }
            FeatureBinner::Categorical(map) => {
                std::iter::once("-1".to_string())
                    .chain((0..map.len()).map(|i| i.to_string()))
                    .collect::<Vec<_>>().join(":")
            }
        }).collect();

        let obj = self.objective.name();
        let mut s = format!("\
tree
version=v4
num_class=1
num_tree_per_iteration=1
label_index=0
max_feature_idx={mfi}
objective={obj}
feature_names={names}
feature_infos={infos}
tree_sizes={sizes}

",
            mfi = self.feature_binners.len() - 1,
            names = self.feature_names.join(" "),
            infos = feature_infos.join(" "),
            sizes = tree_texts.iter().map(|t| t.len().to_string()).collect::<Vec<_>>().join(" "),
        );
        for t in &tree_texts { s.push_str(t); }
        write!(s, "\
end of trees

parameters:
[boosting: gbdt]
[objective: {obj}]
[num_class: 1]
[num_tree_per_iteration: 1]
end of parameters

pandas_categorical:null
").unwrap();
        s
    }
}

/// Write a 1-leaf tree (used to encode the booster's base_score).
fn write_constant_tree(s: &mut String, value: f64) {
    write!(s, "\
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
").unwrap();
}

impl Tree {
    pub(crate) fn write_to_string(&self, s: &mut String) {
        // LightGBM's child indexing: i for internal node i, -(j+1) for leaf j.
        let mut node_idx = vec![0i32; self.nodes.len()];
        let mut leaf_values = Vec::new();
        let mut n_internal = 0i32;
        for (i, node) in self.nodes.iter().enumerate() {
            match node {
                Node::Internal { .. } => { node_idx[i] = n_internal; n_internal += 1; }
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
            let Node::Internal { left_child: lc, right_child: rc, split_feature: sf, missing_goes_left, threshold: t } = node else { continue };
            split_feature.push(*sf);
            // bit 0: categorical, bit 1: default_left, bits 2-3: missing_type (NaN=2)
            let mut d = 8u8;
            if *missing_goes_left { d |= 2; }
            match t {
                Threshold::Numeric(v) => threshold.push(*v),
                Threshold::Categorical(bitset) => {
                    d |= 1;
                    threshold.push((cat_boundaries.len() - 1) as f64);
                    for &word in bitset {
                        cat_thresholds.push(word as u32);
                        cat_thresholds.push((word >> 32) as u32);
                    }
                    cat_boundaries.push(cat_thresholds.len());
                }
            }
            decision_type.push(d);
            left_child.push(node_idx[*lc as usize]);
            right_child.push(node_idx[*rc as usize]);
        }

        fn join<T: std::fmt::Display>(v: &[T]) -> String {
            v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
        }
        let n_int = split_feature.len();
        let n_leaf = leaf_values.len();
        write!(s, "\
num_leaves={n_leaf}
num_cat={n_cat}
split_feature={split_feature}
split_gain={zeros_int}
threshold={threshold}
decision_type={decision_type}
left_child={left_child}
right_child={right_child}
leaf_value={leaf_value}
leaf_weight={zeros_leaf}
leaf_count={zeros_leaf}
internal_value={zeros_int}
internal_weight={zeros_int}
internal_count={zeros_int}
",
            n_cat = cat_boundaries.len() - 1,
            split_feature = join(&split_feature),
            threshold = join(&threshold),
            decision_type = join(&decision_type),
            left_child = join(&left_child),
            right_child = join(&right_child),
            leaf_value = join(&leaf_values),
            zeros_int = join(&vec![0u8; n_int]),
            zeros_leaf = join(&vec![0u8; n_leaf]),
        ).unwrap();
        if !cat_thresholds.is_empty() {
            writeln!(s, "cat_boundaries={}", join(&cat_boundaries)).unwrap();
            writeln!(s, "cat_threshold={}", join(&cat_thresholds)).unwrap();
        }
        s.push_str("is_linear=0\nshrinkage=1\n");
    }
}

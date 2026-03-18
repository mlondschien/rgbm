/// Score of a leaf node used for gain calculation.
///
/// Branchless implementation for optimal SIMD performance. LGBM reference:
/// `src/treelearner/feature_histogram.hpp` `GetLeafSplitGainGivenOutput`.
#[inline(always)]
pub fn calculate_score(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    let d = (g.abs() - l1).max(0.0);
    d * d / (h + l2)
}

/// Optimal leaf value (weight) for a leaf node.
///
/// LGBM reference: `src/treelearner/feature_histogram.hpp` `CalculateSplittedLeafOutput`.
#[inline(always)]
pub fn calculate_weight(g: f64, h: f64, l1: f64, l2: f64) -> f64 {
    let d = (g.abs() - l1).max(0.0);
    -g.signum() * d / (h + l2)
}

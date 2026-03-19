/// Objective function for gradient boosting — computes per-row gradients and hessians.
pub trait Objective: Send + Sync {
    fn gradient(&self, label: f64, score: f64) -> f64;
    fn hessian(&self, label: f64, score: f64) -> f64;
    fn initial_score(&self, labels: &[f64]) -> f64;
    fn prediction(&self, score: f64) -> f64;
}

pub struct SquaredLoss;

impl Objective for SquaredLoss {
    fn gradient(&self, label: f64, score: f64) -> f64 {
        score - label
    }

    fn hessian(&self, _label: f64, _score: f64) -> f64 {
        1.0
    }

    fn initial_score(&self, labels: &[f64]) -> f64 {
        if labels.is_empty() { 0.0 } else { labels.iter().sum::<f64>() / labels.len() as f64 }
    }

    fn prediction(&self, score: f64) -> f64 {
        score
    }
}

/// Binary cross-entropy. Scores are log-odds; labels are in {0, 1}.
pub struct BinaryLogloss;

impl Objective for BinaryLogloss {
    fn gradient(&self, label: f64, score: f64) -> f64 {
        self.prediction(score) - label
    }

    fn hessian(&self, _label: f64, score: f64) -> f64 {
        let p = self.prediction(score);
        (p * (1.0 - p)).max(1e-16)
    }

    fn initial_score(&self, labels: &[f64]) -> f64 {
        if labels.is_empty() { return 0.0; }
        let mean = (labels.iter().sum::<f64>() / labels.len() as f64).clamp(1e-7, 1.0 - 1e-7);
        (mean / (1.0 - mean)).ln()
    }

    fn prediction(&self, score: f64) -> f64 {
        1.0 / (1.0 + (-score).exp())
    }
}

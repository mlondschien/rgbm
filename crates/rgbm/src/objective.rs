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

/// Probit loss. Scores are on the latent normal scale, labels are in {0, 1}.
pub struct Probit;
impl Probit {
    #[inline]
    fn norm_cdf(x: f64) -> f64 {
        0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
    }

    #[inline]
    fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }
}

impl Objective for Probit {
    fn gradient(&self, label: f64, score: f64) -> f64 {
        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
        Self::norm_pdf(score) * (p - label) / (p * (1.0 - p))
    }

    fn hessian(&self, _label: f64, score: f64) -> f64 {
        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
        let phi = Self::norm_pdf(score);
        (phi * phi / (p * (1.0 - p))).max(1e-16)
    }

    /// Inverse cdf not implemented in libm. Possibly todo via newton's method.
    fn initial_score(&self, _labels: &[f64]) -> f64 {
        0.0
    }

    fn prediction(&self, score: f64) -> f64 {
        Self::norm_cdf(score)
    }
}

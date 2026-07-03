// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use rayon::prelude::*;

/// Objective function for gradient boosting — computes per-row gradients and hessians.
pub trait Objective: Send + Sync {
    fn lgbm_name(&self) -> &str;

    /// Per-row gradient and hessian.
    fn grad_hess(&self, label: f64, score: f64, weight: f64) -> [f32; 2];

    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    ) {
        match pool {
            Some(pool) => pool.install(|| {
                out.par_iter_mut().enumerate().for_each(|(i, gh)| {
                    let weight = weights.map_or(1.0, |w| w[i]);
                    *gh = self.grad_hess(labels[i], scores[i], weight);
                });
            }),
            None => {
                for (i, gh) in out.iter_mut().enumerate() {
                    let weight = weights.map_or(1.0, |w| w[i]);
                    *gh = self.grad_hess(labels[i], scores[i], weight);
                }
            }
        }
    }

    fn initial_score(&self, labels: &[f64], weights: Option<&[f64]>) -> f64;
    fn prediction(&self, score: f64) -> f64;
}

pub struct Gaussian;

impl Objective for Gaussian {
    fn lgbm_name(&self) -> &str {
        "regression"
    }
    fn grad_hess(&self, label: f64, score: f64, weight: f64) -> [f32; 2] {
        [((score - label) * weight) as f32, weight as f32]
    }

    fn initial_score(&self, labels: &[f64], weights: Option<&[f64]>) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        match weights {
            Some(weights) => {
                let (sum_wy, sum_w) = labels
                    .iter()
                    .zip(weights.iter())
                    .fold((0.0, 0.0), |(swy, sw), (&y, &w)| (swy + w * y, sw + w));
                if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 }
            }
            None => labels.iter().sum::<f64>() / labels.len() as f64,
        }
    }

    fn prediction(&self, score: f64) -> f64 {
        score
    }
}

/// Binary cross-entropy. Scores are log-odds; labels are in {0, 1}.
pub struct Logistic;

impl Objective for Logistic {
    fn lgbm_name(&self) -> &str {
        "binary"
    }

    fn grad_hess(&self, label: f64, score: f64, weight: f64) -> [f32; 2] {
        let p = 1.0 / (1.0 + (-score).exp());
        [
            ((p - label) * weight) as f32,
            ((p * (1.0 - p)).max(1e-16) * weight) as f32,
        ]
    }

    fn initial_score(&self, labels: &[f64], weights: Option<&[f64]>) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        let mean = match weights {
            Some(weights) => {
                let (sum_wy, sum_w) = labels
                    .iter()
                    .zip(weights.iter())
                    .fold((0.0, 0.0), |(swy, sw), (&y, &w)| (swy + w * y, sw + w));
                if sum_w > 0.0 { sum_wy / sum_w } else { 0.5 }
            }
            None => labels.iter().sum::<f64>() / labels.len() as f64,
        };
        let mean = mean.clamp(1e-7, 1.0 - 1e-7);
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
    fn lgbm_name(&self) -> &str {
        "binary"
    }

    fn grad_hess(&self, label: f64, score: f64, weight: f64) -> [f32; 2] {
        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
        let phi = Self::norm_pdf(score);
        let v = p * (1.0 - p);
        [
            (phi * (p - label) / v * weight) as f32,
            (phi * phi / v * weight).max(1e-16) as f32,
        ]
    }

    /// Inverse cdf not implemented in libm. Possibly todo via newton's method.
    fn initial_score(&self, _labels: &[f64], _weights: Option<&[f64]>) -> f64 {
        0.0
    }

    fn prediction(&self, score: f64) -> f64 {
        Self::norm_cdf(score)
    }
}

/// Poisson regression. Scores are log-rates: ``predict = exp(score)``.
/// Labels are non-negative counts.
pub struct Poisson;

impl Objective for Poisson {
    fn lgbm_name(&self) -> &str {
        "poisson"
    }

    fn grad_hess(&self, label: f64, score: f64, weight: f64) -> [f32; 2] {
        let lambda = score.exp().min(1e30);
        [
            ((lambda - label) * weight) as f32,
            (lambda * weight).max(1e-16) as f32,
        ]
    }

    fn initial_score(&self, labels: &[f64], weights: Option<&[f64]>) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }
        let mean = match weights {
            Some(weights) => {
                let (sum_wy, sum_w) = labels
                    .iter()
                    .zip(weights.iter())
                    .fold((0.0, 0.0), |(swy, sw), (&y, &w)| (swy + w * y, sw + w));
                if sum_w > 0.0 { sum_wy / sum_w } else { 0.0 }
            }
            None => labels.iter().sum::<f64>() / labels.len() as f64,
        };
        mean.max(1e-10).ln()
    }

    fn prediction(&self, score: f64) -> f64 {
        score.exp()
    }
}

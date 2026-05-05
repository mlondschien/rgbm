// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use rayon::prelude::*;

/// Objective function for gradient boosting — computes per-row gradients and hessians.
pub trait Objective: Send + Sync {
    fn gradient_hessian(&self, labels: &[f64], scores: &[f64], out: &mut [[f32; 2]], pool: Option<&rayon::ThreadPool>);
    fn initial_score(&self, labels: &[f64]) -> f64;
    fn prediction(&self, score: f64) -> f64;
}

pub struct SquaredLoss;

impl Objective for SquaredLoss {
    fn gradient_hessian(&self, labels: &[f64], scores: &[f64], out: &mut [[f32; 2]], pool: Option<&rayon::ThreadPool>) {
        match pool {
            Some(pool) => pool.install(|| {
                out.par_iter_mut().zip(labels.par_iter()).zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        gh[0] = (score - label) as f32;
                        gh[1] = 1.0;
                    });
            }),
            None => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    gh[0] = (score - label) as f32;
                    gh[1] = 1.0;
                }
            }
        }
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
    fn gradient_hessian(&self, labels: &[f64], scores: &[f64], out: &mut [[f32; 2]], pool: Option<&rayon::ThreadPool>) {
        match pool {
            Some(pool) => pool.install(|| {
                out.par_iter_mut().zip(labels.par_iter()).zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        let p = 1.0 / (1.0 + (-score).exp());
                        gh[0] = (p - label) as f32;
                        gh[1] = (p * (1.0 - p)).max(1e-16) as f32;
                    });
            }),
            None => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    let p = 1.0 / (1.0 + (-score).exp());
                    gh[0] = (p - label) as f32;
                    gh[1] = (p * (1.0 - p)).max(1e-16) as f32;
                }
            }
        }
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
    fn gradient_hessian(&self, labels: &[f64], scores: &[f64], out: &mut [[f32; 2]], pool: Option<&rayon::ThreadPool>) {
        match pool {
            Some(pool) => pool.install(|| {
                out.par_iter_mut().zip(labels.par_iter()).zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
                        let phi = Self::norm_pdf(score);
                        let v = p * (1.0 - p);
                        gh[0] = (phi * (p - label) / v) as f32;
                        gh[1] = (phi * phi / v).max(1e-16) as f32;
                    });
            }),
            None => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
                    let phi = Self::norm_pdf(score);
                    let v = p * (1.0 - p);
                    gh[0] = (phi * (p - label) / v) as f32;
                    gh[1] = (phi * phi / v).max(1e-16) as f32;
                }
            }
        }
    }

    /// Inverse cdf not implemented in libm. Possibly todo via newton's method.
    fn initial_score(&self, _labels: &[f64]) -> f64 {
        0.0
    }

    fn prediction(&self, score: f64) -> f64 {
        Self::norm_cdf(score)
    }
}

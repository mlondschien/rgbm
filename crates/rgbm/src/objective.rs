// Copyright (c) 2026 Malte Londschien
// SPDX-License-Identifier: BSD-3-Clause

use rayon::prelude::*;

/// Objective function for gradient boosting — computes per-row gradients and hessians.
pub trait Objective: Send + Sync {
    fn lgbm_name(&self) -> &str;
    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    );
    fn initial_score(&self, labels: &[f64], weights: Option<&[f64]>) -> f64;
    fn prediction(&self, score: f64) -> f64;
}

pub struct Gaussian;

impl Objective for Gaussian {
    fn lgbm_name(&self) -> &str {
        "regression"
    }
    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    ) {
        match (pool, weights) {
            (Some(pool), Some(weights)) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .zip(weights.par_iter())
                    .for_each(|(((gh, &label), &score), &weight)| {
                        gh[0] = ((score - label) * weight) as f32;
                        gh[1] = weight as f32;
                    });
            }),
            (Some(pool), None) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        gh[0] = (score - label) as f32;
                        gh[1] = 1.0;
                    });
            }),
            (None, Some(weights)) => {
                for (((gh, &label), &score), &weight) in
                    out.iter_mut().zip(labels).zip(scores).zip(weights)
                {
                    gh[0] = ((score - label) * weight) as f32;
                    gh[1] = weight as f32;
                }
            }
            (None, None) => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    gh[0] = (score - label) as f32;
                    gh[1] = 1.0;
                }
            }
        }
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

    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    ) {
        match (pool, weights) {
            (Some(pool), Some(weights)) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .zip(weights.par_iter())
                    .for_each(|(((gh, &label), &score), &weight)| {
                        let p = 1.0 / (1.0 + (-score).exp());
                        gh[0] = ((p - label) * weight) as f32;
                        gh[1] = ((p * (1.0 - p)).max(1e-16) * weight) as f32;
                    });
            }),
            (Some(pool), None) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        let p = 1.0 / (1.0 + (-score).exp());
                        gh[0] = (p - label) as f32;
                        gh[1] = (p * (1.0 - p)).max(1e-16) as f32;
                    });
            }),
            (None, Some(weights)) => {
                for (((gh, &label), &score), &weight) in
                    out.iter_mut().zip(labels).zip(scores).zip(weights)
                {
                    let p = 1.0 / (1.0 + (-score).exp());
                    gh[0] = ((p - label) * weight) as f32;
                    gh[1] = ((p * (1.0 - p)).max(1e-16) * weight) as f32;
                }
            }
            (None, None) => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    let p = 1.0 / (1.0 + (-score).exp());
                    gh[0] = (p - label) as f32;
                    gh[1] = (p * (1.0 - p)).max(1e-16) as f32;
                }
            }
        }
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

    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    ) {
        match (pool, weights) {
            (Some(pool), Some(weights)) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .zip(weights.par_iter())
                    .for_each(|(((gh, &label), &score), &weight)| {
                        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
                        let phi = Self::norm_pdf(score);
                        let v = p * (1.0 - p);
                        gh[0] = (phi * (p - label) / v * weight) as f32;
                        gh[1] = (phi * phi / v * weight).max(1e-16) as f32;
                    });
            }),
            (Some(pool), None) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
                        let phi = Self::norm_pdf(score);
                        let v = p * (1.0 - p);
                        gh[0] = (phi * (p - label) / v) as f32;
                        gh[1] = (phi * phi / v).max(1e-16) as f32;
                    });
            }),
            (None, Some(weights)) => {
                for (((gh, &label), &score), &weight) in
                    out.iter_mut().zip(labels).zip(scores).zip(weights)
                {
                    let p = Self::norm_cdf(score).clamp(1e-7, 1.0 - 1e-7);
                    let phi = Self::norm_pdf(score);
                    let v = p * (1.0 - p);
                    gh[0] = (phi * (p - label) / v * weight) as f32;
                    gh[1] = (phi * phi / v * weight).max(1e-16) as f32;
                }
            }
            (None, None) => {
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

    fn gradient_hessian(
        &self,
        labels: &[f64],
        scores: &[f64],
        weights: Option<&[f64]>,
        out: &mut [[f32; 2]],
        pool: Option<&rayon::ThreadPool>,
    ) {
        match (pool, weights) {
            (Some(pool), Some(weights)) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .zip(weights.par_iter())
                    .for_each(|(((gh, &label), &score), &weight)| {
                        let lambda = score.exp().min(1e30);
                        gh[0] = ((lambda - label) * weight) as f32;
                        gh[1] = (lambda * weight).max(1e-16) as f32;
                    });
            }),
            (Some(pool), None) => pool.install(|| {
                out.par_iter_mut()
                    .zip(labels.par_iter())
                    .zip(scores.par_iter())
                    .for_each(|((gh, &label), &score)| {
                        let lambda = score.exp().min(1e30);
                        gh[0] = (lambda - label) as f32;
                        gh[1] = lambda.max(1e-16) as f32;
                    });
            }),
            (None, Some(weights)) => {
                for (((gh, &label), &score), &weight) in
                    out.iter_mut().zip(labels).zip(scores).zip(weights)
                {
                    let lambda = score.exp().min(1e30);
                    gh[0] = ((lambda - label) * weight) as f32;
                    gh[1] = (lambda * weight).max(1e-16) as f32;
                }
            }
            (None, None) => {
                for ((gh, &label), &score) in out.iter_mut().zip(labels).zip(scores) {
                    let lambda = score.exp().min(1e30);
                    gh[0] = (lambda - label) as f32;
                    gh[1] = lambda.max(1e-16) as f32;
                }
            }
        }
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

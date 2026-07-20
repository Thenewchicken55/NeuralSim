//! Mutation operators for genomes.
//!
//! Each operator perturbs a specific part of the genome with a configurable
//! probability and magnitude:
//!
//! - **Weight mutation**: Gaussian perturbation of individual synapse weights
//! - **Param mutation**: perturb neuron model parameters (thresholds, taus, etc.)
//! - **Plasticity mutation**: perturb STDP learning rates, homeostatic targets
//! - **Connection mutation**: perturb inter-region probability and weight scale

use super::genome::Genome;
use crate::neuron::NeuronModelParams;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for mutation operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    /// Probability of mutating each individual synapse weight.
    pub weight_mutation_rate: f64,
    /// Standard deviation of Gaussian perturbation applied to weights.
    pub weight_mutation_sigma: f64,
    /// Probability of completely reinitializing a weight (jump mutation).
    pub weight_reset_rate: f64,
    /// Probability of mutating neuron model parameters per region.
    pub param_mutation_rate: f64,
    /// Sigma for parameter perturbation (relative to param magnitude).
    pub param_mutation_sigma: f64,
    /// Probability of mutating plasticity config.
    pub plasticity_mutation_rate: f64,
    /// Sigma for plasticity param perturbation.
    pub plasticity_mutation_sigma: f64,
    /// Probability of mutating connection topology params.
    pub connection_mutation_rate: f64,
    /// Min/max bounds for weights.
    pub weight_min: f64,
    pub weight_max: f64,
}

impl Default for MutationConfig {
    fn default() -> Self {
        Self {
            weight_mutation_rate: 0.1,
            weight_mutation_sigma: 0.3,
            weight_reset_rate: 0.01,
            param_mutation_rate: 0.2,
            param_mutation_sigma: 0.1,
            plasticity_mutation_rate: 0.3,
            plasticity_mutation_sigma: 0.1,
            connection_mutation_rate: 0.15,
            weight_min: 0.0,
            weight_max: 10.0,
        }
    }
}

/// Apply all mutation operators to a genome in-place.
pub fn mutate(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    mutate_weights(genome, config, rng);
    mutate_params(genome, config, rng);
    mutate_plasticity(genome, config, rng);
    mutate_connections(genome, config, rng);
}

/// Perturb individual synapse weights with Gaussian noise.
fn mutate_weights(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    for w in genome.initial_weights.iter_mut() {
        if rng.random::<f64>() < config.weight_reset_rate {
            // Jump mutation: reinitialize to a completely random value
            *w = rng.random::<f64>() * (config.weight_max - config.weight_min) + config.weight_min;
        } else if rng.random::<f64>() < config.weight_mutation_rate {
            // Gaussian perturbation (Box-Muller transform)
            let delta = gaussian_sample(0.0, config.weight_mutation_sigma, rng);
            *w = (*w + delta).clamp(config.weight_min, config.weight_max);
        }
    }
}

/// Perturb neuron model parameters within each region.
fn mutate_params(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    for region in genome.regions.iter_mut() {
        if rng.random::<f64>() > config.param_mutation_rate {
            continue;
        }
        region.model_params =
            perturb_model_params(&region.model_params, config.param_mutation_sigma, rng);
        // Also perturb excitatory ratio slightly
        let delta = (rng.random::<f64>() - 0.5) * config.param_mutation_sigma;
        region.excitatory_ratio = (region.excitatory_ratio + delta).clamp(0.5, 0.95);
    }
}

/// Perturb plasticity configuration parameters.
fn mutate_plasticity(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    if rng.random::<f64>() > config.plasticity_mutation_rate {
        return;
    }
    let sigma = config.plasticity_mutation_sigma;

    if let Some(ref mut stdp) = genome.plasticity.stdp {
        stdp.a_plus *= gaussian_factor(sigma, rng);
        stdp.a_minus *= gaussian_factor(sigma, rng);
        stdp.tau_plus *= gaussian_factor(sigma, rng);
        stdp.tau_minus *= gaussian_factor(sigma, rng);
        // Keep A-/A+ ratio reasonable
        stdp.a_plus = stdp.a_plus.clamp(0.001, 0.1);
        stdp.a_minus = stdp.a_minus.clamp(0.001, 0.1);
        stdp.tau_plus = stdp.tau_plus.clamp(5.0, 100.0);
        stdp.tau_minus = stdp.tau_minus.clamp(5.0, 100.0);
    }

    // Perturb homeostatic target
    genome.plasticity.homeostatic_target_rate *= gaussian_factor(sigma, rng);
    genome.plasticity.homeostatic_target_rate =
        genome.plasticity.homeostatic_target_rate.clamp(1.0, 50.0);
}

/// Perturb inter-region connection parameters (probability, weight scale).
fn mutate_connections(genome: &mut Genome, config: &MutationConfig, rng: &mut impl Rng) {
    for conn in genome.connections.iter_mut() {
        if rng.random::<f64>() > config.connection_mutation_rate {
            continue;
        }
        let delta_p = (rng.random::<f64>() - 0.5) * 0.02;
        conn.probability = (conn.probability + delta_p).clamp(0.001, 0.5);
        conn.weight_scale *= gaussian_factor(config.param_mutation_sigma, rng);
        conn.weight_scale = conn.weight_scale.clamp(0.1, 5.0);
    }
}

/// Perturb a `NeuronModelParams` by perturbing each field with relative Gaussian noise.
fn perturb_model_params(
    params: &NeuronModelParams,
    sigma: f64,
    rng: &mut impl Rng,
) -> NeuronModelParams {
    match params {
        NeuronModelParams::Lif {
            resting,
            threshold,
            reset,
            tau_m,
            refractory_period,
            input_resistance,
        } => NeuronModelParams::Lif {
            resting: resting + gaussian_noise(sigma, rng),
            threshold: threshold + gaussian_noise(sigma, rng),
            reset: reset + gaussian_noise(sigma, rng),
            tau_m: (*tau_m * gaussian_factor(sigma, rng)).max(0.5),
            refractory_period: (*refractory_period * gaussian_factor(sigma, rng)).max(0.1),
            input_resistance: (*input_resistance * gaussian_factor(sigma, rng)).max(0.1),
        },
        NeuronModelParams::Izhikevich { a, b, c, d } => NeuronModelParams::Izhikevich {
            a: (*a * gaussian_factor(sigma, rng)).max(0.001),
            b: (*b * gaussian_factor(sigma, rng)).max(0.001),
            c: *c + gaussian_noise(sigma * 5.0, rng),
            d: *d * gaussian_factor(sigma, rng),
        },
        NeuronModelParams::HodgkinHuxley {
            g_na,
            g_k,
            g_l,
            e_na,
            e_k,
            e_l,
            c_m,
        } => NeuronModelParams::HodgkinHuxley {
            g_na: *g_na * gaussian_factor(sigma, rng),
            g_k: *g_k * gaussian_factor(sigma, rng),
            g_l: *g_l * gaussian_factor(sigma, rng),
            e_na: *e_na + gaussian_noise(sigma * 5.0, rng),
            e_k: *e_k + gaussian_noise(sigma * 5.0, rng),
            e_l: *e_l + gaussian_noise(sigma * 5.0, rng),
            c_m: (*c_m * gaussian_factor(sigma, rng)).max(0.1),
        },
    }
}

/// Multiplicative Gaussian factor centered at 1.0: `exp(N(0, sigma))`.
/// Ensures parameters stay positive when multiplied.
fn gaussian_factor(sigma: f64, rng: &mut impl Rng) -> f64 {
    gaussian_sample(0.0, sigma, rng).exp()
}

/// Additive Gaussian noise: `N(0, sigma)`.
fn gaussian_noise(sigma: f64, rng: &mut impl Rng) -> f64 {
    gaussian_sample(0.0, sigma, rng)
}

/// Sample from a Gaussian distribution using the Box-Muller transform.
/// This avoids the need for an external `rand_distr` dependency that may
/// conflict with the project's `rand` version.
fn gaussian_sample(mean: f64, sigma: f64, rng: &mut impl Rng) -> f64 {
    let u1 = rng.random::<f64>().max(1e-10);
    let u2 = rng.random::<f64>();
    let z0 = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
    mean + sigma * z0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mutation_changes_some_weights() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng);
        let original: Vec<f64> = genome.initial_weights.clone();
        let config = MutationConfig::default();
        mutate(&mut genome, &config, &mut rng);
        let changed = original
            .iter()
            .zip(genome.initial_weights.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-10)
            .count();
        assert!(changed > 0, "Mutation should change at least some weights");
    }

    #[test]
    fn test_weights_stay_bounded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng);
        let config = MutationConfig {
            weight_mutation_rate: 1.0,
            weight_mutation_sigma: 100.0,
            weight_reset_rate: 0.5,
            ..Default::default()
        };
        mutate(&mut genome, &config, &mut rng);
        for w in &genome.initial_weights {
            assert!(
                *w >= config.weight_min && *w <= config.weight_max,
                "Weight {} out of bounds [{}, {}]",
                w,
                config.weight_min,
                config.weight_max
            );
        }
    }

    #[test]
    fn test_param_mutation_preserves_variants() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng);
        let config = MutationConfig {
            param_mutation_rate: 1.0,
            ..Default::default()
        };
        mutate(&mut genome, &config, &mut rng);
        // After mutation, params should still be valid (the build should succeed)
        let engine = genome.build_engine();
        assert!(engine.network.read().neuron_count() > 0);
    }

    #[test]
    fn test_no_mutation_when_rates_zero() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng);
        let original = genome.clone();
        let config = MutationConfig {
            weight_mutation_rate: 0.0,
            weight_reset_rate: 0.0,
            param_mutation_rate: 0.0,
            plasticity_mutation_rate: 0.0,
            connection_mutation_rate: 0.0,
            ..Default::default()
        };
        mutate(&mut genome, &config, &mut rng);
        assert_eq!(genome.initial_weights, original.initial_weights);
    }
}

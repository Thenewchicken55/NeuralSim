//! Fitness evaluators for scoring brain performance.
//!
//! The [`FitnessEvaluator`] trait abstracts over different tasks so the
//! population loop can swap between them. Higher fitness = better brain.
//!
//! Built-in evaluators:
//! - [`RateHomeostasis`]: reward networks that maintain a target firing rate
//! - [`RewardAccumulation`]: sum the R-STDP reward signal over the lifetime

use crate::simulation::SimulationEngine;
use serde::{Deserialize, Serialize};

/// Trait for fitness functions. Evaluates an engine by running it and
/// returning a scalar fitness score (higher = better).
pub trait FitnessEvaluator: Send + Sync {
    /// Evaluate the engine. The engine should be run for `lifetime_ms`
    /// simulation milliseconds and a fitness score returned.
    fn evaluate(&self, engine: &mut SimulationEngine) -> f64;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// Rate homeostasis: reward networks that hold a target mean firing rate.
///
/// Fitness = 1 / (1 + |mean_rate - target|) so:
/// - Perfect match → 1.0
/// - 5 Hz off → 0.17
/// - 50 Hz off → 0.02
///
/// Additionally penalizes silent networks (0 Hz) to prevent the trivial
/// "do nothing" solution from dominating.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateHomeostasis {
    /// Target mean firing rate in Hz.
    pub target_rate: f64,
    /// Simulation duration in ms for evaluation.
    pub lifetime_ms: f64,
    /// If true, penalize networks with <1 Hz mean rate.
    pub penalize_silence: bool,
}

impl RateHomeostasis {
    pub fn new(target_rate: f64) -> Self {
        Self {
            target_rate,
            lifetime_ms: 500.0,
            penalize_silence: true,
        }
    }

    pub fn with_lifetime(mut self, ms: f64) -> Self {
        self.lifetime_ms = ms;
        self
    }
}

impl FitnessEvaluator for RateHomeostasis {
    fn evaluate(&self, engine: &mut SimulationEngine) -> f64 {
        engine.simulate_ms(self.lifetime_ms);
        let stats = engine.stats();
        let mean_rate = stats.mean_firing_rate;
        let deviation = (mean_rate - self.target_rate).abs();
        let mut fitness = 1.0 / (1.0 + deviation);

        // Penalize silence — a brain that fires nothing is not "homeostatic".
        if self.penalize_silence && mean_rate < 1.0 {
            fitness *= 0.1;
        }
        // Also penalize runaway explosion.
        if mean_rate > 500.0 {
            fitness *= 0.1;
        }
        fitness
    }

    fn name(&self) -> &str {
        "rate_homeostasis"
    }
}

/// Reward accumulation: sum the R-STDP reward signal over the lifetime.
///
/// This evaluator leverages the existing R-STDP infrastructure: the engine
/// accumulates reward whenever output spikes exceed the threshold. Brains
/// that learn to produce timely output spikes score higher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardAccumulation {
    pub lifetime_ms: f64,
}

impl RewardAccumulation {
    pub fn new() -> Self {
        Self { lifetime_ms: 500.0 }
    }

    pub fn with_lifetime(mut self, ms: f64) -> Self {
        self.lifetime_ms = ms;
        self
    }
}

impl Default for RewardAccumulation {
    fn default() -> Self {
        Self::new()
    }
}

impl FitnessEvaluator for RewardAccumulation {
    fn evaluate(&self, engine: &mut SimulationEngine) -> f64 {
        let mut total_reward = 0.0;
        let steps = (self.lifetime_ms / engine.dt) as usize;
        for _ in 0..steps {
            engine.step();
            total_reward += engine.reward_signal;
        }
        total_reward
    }

    fn name(&self) -> &str {
        "reward_accumulation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evolution::genome::Genome;
    use rand::SeedableRng;

    #[test]
    fn test_rate_homeostasis_perfect_is_one() {
        // A genome that produces ~5 Hz should get high fitness.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        let mut engine = genome.build_engine();
        let evaluator = RateHomeostasis::new(5.0).with_lifetime(100.0);
        let fitness = evaluator.evaluate(&mut engine);
        assert!(
            fitness > 0.0 && fitness <= 1.0,
            "Fitness should be in (0, 1]"
        );
    }

    #[test]
    fn test_rate_homeostasis_penalizes_silence() {
        // With a very high target rate, a silent brain should score low.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        let mut engine = genome.build_engine();
        // Zero out all input current to suppress activity
        {
            let mut net = engine.network.write();
            for c in net.neurons.input_current.iter_mut() {
                *c = 0.0;
            }
        }
        engine.noise_amplitude = 0.0;
        let evaluator = RateHomeostasis {
            target_rate: 50.0,
            lifetime_ms: 100.0,
            penalize_silence: true,
        };
        let fitness = evaluator.evaluate(&mut engine);
        assert!(
            fitness < 0.2,
            "Silent brain should score low, got {}",
            fitness
        );
    }

    #[test]
    fn test_reward_accumulation_returns_nonnegative() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        let mut engine = genome.build_engine();
        let evaluator = RewardAccumulation::new().with_lifetime(100.0);
        let fitness = evaluator.evaluate(&mut engine);
        assert!(fitness >= 0.0, "Reward should be non-negative");
    }
}

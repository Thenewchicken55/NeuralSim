//! Crossover (recombination) operators for genomes.
//!
//! Two modes are provided:
//! - [`crossover_uniform`]: for each gene, randomly select from parent A or B.
//! - [`crossover_blx_alpha`]: BLX-α blending — child gene is sampled from an
//!   extended interval between the two parents. Better for real-valued GAs.

use super::genome::Genome;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Which crossover strategy to use.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CrossoverMode {
    Uniform,
    BlxAlpha,
}

/// Uniform crossover: each gene is independently inherited from one of the
/// two parents with equal probability. Topology is inherited wholesale from
/// parent A (since both parents share the same topology in fixed-topology EA).
pub fn crossover_uniform(parent_a: &Genome, parent_b: &Genome, rng: &mut impl Rng) -> Genome {
    let len = parent_a
        .initial_weights
        .len()
        .min(parent_b.initial_weights.len());
    let initial_weights: Vec<f64> = (0..len)
        .map(|i| {
            if rng.random::<f64>() < 0.5 {
                parent_a.initial_weights[i]
            } else {
                parent_b.initial_weights[i]
            }
        })
        .collect();

    // Inherit topology from parent A (same topology expected in fixed-topology EA).
    // For region params and plasticity, pick from one parent randomly.
    let regions: Vec<_> = parent_a
        .regions
        .iter()
        .zip(parent_b.regions.iter())
        .map(|(ra, rb)| {
            if rng.random::<f64>() < 0.5 {
                ra.clone()
            } else {
                rb.clone()
            }
        })
        .collect();

    let connections: Vec<_> = parent_a
        .connections
        .iter()
        .zip(parent_b.connections.iter())
        .map(|(ca, cb)| {
            if rng.random::<f64>() < 0.5 {
                ca.clone()
            } else {
                cb.clone()
            }
        })
        .collect();

    let plasticity = if rng.random::<f64>() < 0.5 {
        parent_a.plasticity.clone()
    } else {
        parent_b.plasticity.clone()
    };

    let seed = if rng.random::<f64>() < 0.5 {
        parent_a.seed
    } else {
        parent_b.seed
    };

    Genome {
        name: format!("{}_x_{}", parent_a.name, parent_b.name),
        regions,
        connections,
        initial_weights,
        plasticity,
        seed,
    }
}

/// BLX-α (blend crossover): for each weight gene, sample uniformly from the
/// interval `[min - α·range, max + α·range]` where `min`/`max` are the two
/// parents' values and `range = max - min`.
///
/// `α = 0.5` is a common default that allows exploration slightly beyond
/// the parental range while staying biased toward it.
pub fn crossover_blx_alpha(
    parent_a: &Genome,
    parent_b: &Genome,
    alpha: f64,
    rng: &mut impl Rng,
) -> Genome {
    let len = parent_a
        .initial_weights
        .len()
        .min(parent_b.initial_weights.len());
    let initial_weights: Vec<f64> = (0..len)
        .map(|i| {
            let (wa, wb) = (parent_a.initial_weights[i], parent_b.initial_weights[i]);
            let lo = wa.min(wb);
            let hi = wa.max(wb);
            let range = hi - lo;
            let lo_ext = lo - alpha * range;
            let hi_ext = hi + alpha * range;
            rng.random::<f64>() * (hi_ext - lo_ext) + lo_ext
        })
        .collect();

    // Topology and discrete params from uniform crossover
    let regions: Vec<_> = parent_a
        .regions
        .iter()
        .zip(parent_b.regions.iter())
        .map(|(ra, rb)| {
            if rng.random::<f64>() < 0.5 {
                ra.clone()
            } else {
                rb.clone()
            }
        })
        .collect();

    let connections: Vec<_> = parent_a
        .connections
        .iter()
        .zip(parent_b.connections.iter())
        .map(|(ca, cb)| {
            if rng.random::<f64>() < 0.5 {
                ca.clone()
            } else {
                cb.clone()
            }
        })
        .collect();

    let plasticity = if rng.random::<f64>() < 0.5 {
        parent_a.plasticity.clone()
    } else {
        parent_b.plasticity.clone()
    };

    let seed = if rng.random::<f64>() < 0.5 {
        parent_a.seed
    } else {
        parent_b.seed
    };

    Genome {
        name: format!("{}_x_{}", parent_a.name, parent_b.name),
        regions,
        connections,
        initial_weights,
        plasticity,
        seed,
    }
}

/// Dispatch to the selected crossover mode.
pub fn crossover(
    mode: CrossoverMode,
    parent_a: &Genome,
    parent_b: &Genome,
    rng: &mut impl Rng,
) -> Genome {
    match mode {
        CrossoverMode::Uniform => crossover_uniform(parent_a, parent_b, rng),
        CrossoverMode::BlxAlpha => crossover_blx_alpha(parent_a, parent_b, 0.5, rng),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_uniform_crossover_produces_valid_genome() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let parent_a = Genome::random(&mut rng);
        let parent_b = Genome::random(&mut rng);
        let child = crossover_uniform(&parent_a, &parent_b, &mut rng);
        assert_eq!(child.initial_weights.len(), parent_a.initial_weights.len());
        // Each child weight should come from one of the parents
        for (i, w) in child.initial_weights.iter().enumerate() {
            assert!(
                (*w - parent_a.initial_weights[i]).abs() < 1e-10
                    || (*w - parent_b.initial_weights[i]).abs() < 1e-10,
                "Child weight should come from a parent"
            );
        }
    }

    #[test]
    fn test_blx_alpha_crossover_blends() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let parent_a = Genome::random(&mut rng);
        let parent_b = Genome::random(&mut rng);
        let child = crossover_blx_alpha(&parent_a, &parent_b, 0.5, &mut rng);
        assert_eq!(child.initial_weights.len(), parent_a.initial_weights.len());
        // Child weights should be valid (not NaN)
        for w in &child.initial_weights {
            assert!(w.is_finite(), "BLX-α should produce finite weights");
        }
    }

    #[test]
    fn test_crossover_preserves_topology() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let parent_a = Genome::random(&mut rng);
        let parent_b = Genome::random(&mut rng);
        let child = crossover(CrossoverMode::Uniform, &parent_a, &parent_b, &mut rng);
        assert_eq!(child.regions.len(), parent_a.regions.len());
        assert_eq!(child.connections.len(), parent_a.connections.len());
    }
}

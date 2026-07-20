//! Evolutionary optimization for spiking neural networks.
//!
//! Implements generations, descent-with-modification, and selection on
//! populations of brains. Each individual's genome encodes the network
//! topology, initial weights, neuron parameters, and plasticity configuration.
//!
//! # Quick start
//!
//! ```no_run
//! use neural_sim::evolution::{Population, RateHomeostasis, EvolutionConfig};
//!
//! let config = EvolutionConfig::default();
//! let mut pop = Population::random(30, &config);
//! let evaluator = RateHomeostasis::new(5.0); // target 5 Hz
//!
//! for gen in 0..100 {
//!     let best = pop.evolve_generation(&evaluator, &config);
//!     println!("Gen {}: best fitness = {:.4}", gen, best);
//! }
//! ```
//!
//! # Design
//!
//! - **Genome**: serializable spec (regions + connections + weights + plasticity).
//!   Built into a `Network` + `SimulationEngine` for evaluation.
//! - **Mutation**: Gaussian perturbation of weights and parameters.
//! - **Crossover**: uniform or BLX-α recombination of two parents.
//! - **Selection**: tournament selection with elitism.
//! - **Fitness**: pluggable via the `FitnessEvaluator` trait.
//!   Built-in: `RateHomeostasis` (reward holding a target firing rate).
//! - **Lamarckian mode**: optionally write learned weights back into the genome
//!   so lifetime plasticity is inherited by offspring.

pub mod crossover;
pub mod fitness;
pub mod genome;
pub mod mutation;
pub mod population;
pub mod selection;

pub use crossover::{CrossoverMode, crossover_blx_alpha, crossover_uniform};
pub use fitness::{FitnessEvaluator, RateHomeostasis, RewardAccumulation};
pub use genome::{ConnectionSpec, Genome, RegionSpec};
pub use mutation::{MutationConfig, mutate};
pub use population::{EvolutionConfig, GenerationStats, Population};
pub use selection::tournament_select;

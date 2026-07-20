//! Genome: a serializable specification for building a brain.
//!
//! A `Genome` captures everything needed to reconstruct an individual:
//! - Region layout (names, neuron counts, model params, excitatory ratios)
//! - Inter-region connections (which regions project to which, probabilities)
//! - Per-synapse initial weights (heritable)
//! - Plasticity configuration (STDP rates, homeostatic targets, etc.)
//!
//! The genome is built into a `Network` + `SimulationEngine` for evaluation.
//! After simulation, learned weights can be extracted back into the genome
//! (Lamarckian inheritance).

use crate::network::Network;
use crate::network::builder::BrainBuilder;
use crate::neuron::NeuronModelParams;
use crate::simulation::SimulationEngine;
use crate::synapse::PlasticityConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Specification for a single brain region in a genome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RegionSpec {
    pub name: String,
    pub neuron_count: usize,
    pub excitatory_ratio: f64,
    pub model_params: NeuronModelParams,
    pub is_input: bool,
    pub is_output: bool,
    /// If true, build this region as a 6-layer cortical column.
    pub is_cortical_column: bool,
}

/// Specification for a connection between two regions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionSpec {
    pub from: String,
    pub to: String,
    pub probability: f64,
    pub weight_scale: f64,
}

/// A complete heritable brain specification.
///
/// The genome is the unit of selection — it encodes the "DNA" of a brain.
/// Topology is determined by `regions` + `connections` (built deterministically
/// with a fixed seed), while `initial_weights` stores the per-synapse weight
/// values that actually evolve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub name: String,
    pub regions: Vec<RegionSpec>,
    pub connections: Vec<ConnectionSpec>,
    /// Per-synapse weights, in the order `BrainBuilder::build()` creates them.
    /// These override the builder's random weights after building.
    pub initial_weights: Vec<f64>,
    /// Plasticity configuration inherited by this individual.
    pub plasticity: PlasticityConfig,
    /// RNG seed for this individual's evaluation.
    pub seed: u64,
}

impl Genome {
    /// Create a random genome with a default cortical-column layout.
    ///
    /// The topology is built deterministically from the region/connection specs,
    /// then random weights are generated for every synapse.
    pub fn random(rng: &mut impl Rng) -> Self {
        let izh = NeuronModelParams::Izhikevich {
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
        };

        let regions = vec![
            RegionSpec {
                name: "Input".into(),
                neuron_count: 100,
                excitatory_ratio: 0.80,
                model_params: izh,
                is_input: true,
                is_output: false,
                is_cortical_column: false,
            },
            RegionSpec {
                name: "V1".into(),
                neuron_count: 500,
                excitatory_ratio: 0.80,
                model_params: NeuronModelParams::default(),
                is_input: false,
                is_output: false,
                is_cortical_column: true,
            },
            RegionSpec {
                name: "Output".into(),
                neuron_count: 100,
                excitatory_ratio: 0.80,
                model_params: izh,
                is_input: false,
                is_output: true,
                is_cortical_column: false,
            },
        ];

        let connections = vec![
            ConnectionSpec {
                from: "Input".into(),
                to: "V1".into(),
                probability: 0.03,
                weight_scale: 1.0,
            },
            ConnectionSpec {
                from: "V1".into(),
                to: "Output".into(),
                probability: 0.03,
                weight_scale: 1.0,
            },
        ];

        // Build once to discover how many synapses the topology produces.
        let scratch = Self::build_network_raw(&regions, &connections);
        let n_syn = scratch.synapse_count();
        let initial_weights: Vec<f64> = (0..n_syn)
            .map(|_| rng.random::<f64>() * 2.0 + 0.5)
            .collect();

        Self {
            name: format!("Genome_{}", rng.random::<u32>()),
            regions,
            connections,
            initial_weights,
            plasticity: PlasticityConfig::default(),
            seed: rng.random(),
        }
    }

    /// Build a `Network` from the genome's topology spec, then apply
    /// the heritable initial weights.
    pub fn build_network(&self) -> Network {
        let mut net = Self::build_network_raw(&self.regions, &self.connections);
        // Override the builder's random weights with our heritable weights.
        let n = net.synapse_count().min(self.initial_weights.len());
        for i in 0..n {
            net.synapses[i].weight = self.initial_weights[i];
        }
        net
    }

    /// Build a full `SimulationEngine` from the genome, with plasticity enabled.
    pub fn build_engine(&self) -> SimulationEngine {
        let network = self.build_network();
        let engine = SimulationEngine::new(network)
            .with_seed(self.seed)
            .with_plasticity(self.plasticity.clone())
            .with_noise(10.0)
            .with_conductance(true);

        // Enable plasticity on all synapses.
        {
            let mut net = engine.network.write();
            for syn in net.synapses.iter_mut() {
                if engine.plasticity.enabled {
                    *syn = syn.clone().with_plasticity();
                }
            }
        }
        engine
    }

    /// Extract a genome from a simulated network (for Lamarckian inheritance).
    ///
    /// The topology is copied from `template` and the weights are read from
    /// the network's current (possibly learned) synapse state.
    pub fn extract_from_network(template: &Genome, network: &Network) -> Self {
        let initial_weights: Vec<f64> = network.synapses.iter().map(|s| s.weight).collect();
        Self {
            name: format!("{}_offspring", template.name),
            regions: template.regions.clone(),
            connections: template.connections.clone(),
            initial_weights,
            plasticity: template.plasticity.clone(),
            seed: template.seed,
        }
    }

    /// Build the topology deterministically from region/connection specs.
    /// Uses a fixed seed (42) so the same specs always produce the same
    /// adjacency structure — only the weights differ between individuals.
    fn build_network_raw(regions: &[RegionSpec], connections: &[ConnectionSpec]) -> Network {
        let mut builder = BrainBuilder::new()
            .with_name("EvolvedBrain")
            .with_plasticity(true);

        for r in regions {
            if r.is_cortical_column {
                builder = builder.add_cortical_column(&r.name, r.neuron_count);
            } else {
                builder =
                    builder.add_region(&r.name, r.neuron_count, r.excitatory_ratio, r.model_params);
            }
            if r.is_input {
                builder = builder.mark_input(&r.name);
            }
            if r.is_output {
                builder = builder.mark_output(&r.name);
            }
        }

        for c in connections {
            builder = builder.connect_regions(&c.from, &c.to, c.probability, c.weight_scale, None);
        }

        builder.build()
    }

    /// Number of neurons in the built network.
    pub fn neuron_count(&self) -> usize {
        self.regions.iter().map(|r| r.neuron_count).sum()
    }

    /// Number of synapses (inferred from initial_weights length).
    pub fn synapse_count(&self) -> usize {
        self.initial_weights.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_random_genome_has_weights() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        assert!(
            !genome.initial_weights.is_empty(),
            "Genome should have synapse weights"
        );
        assert_eq!(genome.regions.len(), 3);
        assert_eq!(genome.connections.len(), 2);
    }

    #[test]
    fn test_build_network_applies_weights() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let mut genome = Genome::random(&mut rng);
        // Set all weights to a recognizable value
        genome.initial_weights.iter_mut().for_each(|w| *w = 2.71);
        let net = genome.build_network();
        for syn in &net.synapses {
            assert!(
                (syn.weight - 2.71).abs() < 1e-10,
                "Weight should be overridden by genome"
            );
        }
    }

    #[test]
    fn test_extract_from_network_roundtrip() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let genome = Genome::random(&mut rng);
        let net = genome.build_network();
        // Modify a weight to simulate learning
        let mut net = net;
        net.synapses[0].weight = 42.0;
        let extracted = Genome::extract_from_network(&genome, &net);
        assert!((extracted.initial_weights[0] - 42.0).abs() < 1e-10);
        // Topology should be preserved
        assert_eq!(extracted.regions, genome.regions);
        assert_eq!(extracted.connections, genome.connections);
    }

    #[test]
    fn test_build_engine_produces_spikes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        let mut engine = genome.build_engine();
        engine.simulate_ms(200.0);
        let stats = engine.stats();
        assert!(
            stats.total_spikes > 0,
            "Engine built from genome should produce spikes"
        );
    }

    #[test]
    fn test_genome_is_serializable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        let json = serde_json::to_string(&genome).expect("serialize");
        let restored: Genome = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.regions.len(), genome.regions.len());
        // Compare weights with tolerance for f64 JSON round-trip precision
        assert_eq!(restored.initial_weights.len(), genome.initial_weights.len());
        for (a, b) in restored
            .initial_weights
            .iter()
            .zip(genome.initial_weights.iter())
        {
            assert!((a - b).abs() < 1e-10, "Weight mismatch after serialization");
        }
        assert_eq!(restored.seed, genome.seed);
    }
}

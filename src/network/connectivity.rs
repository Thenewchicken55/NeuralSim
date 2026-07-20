use crate::network::Network;
use crate::synapse::{SynapseState, SynapseType};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Connectivity pattern configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConnectivityPattern {
    /// Random (Erdos-Renyi) with connection probability p
    ErdosRenyi { p: f64 },
    /// Small-world (Watts-Strogatz) with k nearest neighbors and rewiring probability beta
    WattsStrogatz { k: usize, beta: f64 },
    /// Distance-based with Gaussian profile
    DistanceBased { sigma: f64, max_distance: f64 },
}

impl ConnectivityPattern {
    pub fn connect(&self, network: &mut Network, rng: &mut impl Rng) {
        match self {
            Self::ErdosRenyi { p } => Self::connect_erdos_renyi(network, *p, rng),
            Self::WattsStrogatz { k, beta } => {
                Self::connect_watts_strogatz(network, *k, *beta, rng)
            }
            Self::DistanceBased {
                sigma,
                max_distance,
            } => Self::connect_distance_based(network, *sigma, *max_distance, rng),
        }
        network.finalize();
    }

    fn connect_erdos_renyi(network: &mut Network, p: f64, rng: &mut impl Rng) {
        let n = network.neuron_count();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                if rng.random::<f64>() < p {
                    let syn_type = Self::synapse_type(network, i);
                    network.add_synapse(SynapseState::new(
                        i,
                        j,
                        rng.random::<f64>() * 2.0,
                        syn_type,
                    ));
                }
            }
        }
    }

    fn connect_watts_strogatz(network: &mut Network, k: usize, beta: f64, rng: &mut impl Rng) {
        let n = network.neuron_count();
        let half_k = k / 2;
        for i in 0..n {
            for j in 1..=half_k {
                let target = (i + j) % n;
                let syn_type = Self::synapse_type(network, i);
                network.add_synapse(SynapseState::new(i, target, 1.0, syn_type));

                // Rewire with probability beta
                if rng.random::<f64>() < beta {
                    let new_target = rng.random_range(0..n);
                    if new_target != i {
                        network.add_synapse(SynapseState::new(i, new_target, 1.0, syn_type));
                    }
                }
            }
        }
    }

    fn connect_distance_based(
        network: &mut Network,
        sigma: f64,
        max_dist: f64,
        rng: &mut impl Rng,
    ) {
        let n = network.neuron_count();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist = (i as f64 - j as f64).abs();
                if dist > max_dist {
                    continue;
                }
                let prob = f64::exp(-(dist * dist) / (2.0 * sigma * sigma));
                if rng.random::<f64>() < prob {
                    let syn_type = Self::synapse_type(network, i);
                    network.add_synapse(SynapseState::new(i, j, 1.0, syn_type));
                }
            }
        }
    }

    fn synapse_type(network: &Network, source: usize) -> SynapseType {
        match network.neurons.neuron_type.get(source) {
            Some(crate::neuron::NeuronType::Excitatory) => SynapseType::AMPA,
            Some(crate::neuron::NeuronType::Inhibitory) => SynapseType::GABA,
            None => SynapseType::AMPA,
        }
    }
}

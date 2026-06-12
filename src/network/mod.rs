pub mod graph;
pub mod region;
pub mod connectivity;

use crate::neuron::{NeuronArray, NeuronType};
use crate::synapse::SynapseState;
use serde::{Deserialize, Serialize};

/// The main network container. Holds all neurons and synapses
/// with cache-friendly memory layout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub name: String,
    pub neurons: NeuronArray,
    /// CSR adjacency: for each neuron, range into `synapses`
    pub adjacency_ptr: Vec<usize>,
    /// Flat list of all synapses
    pub synapses: Vec<SynapseState>,
    /// Synapse indices for each source neuron (CSR column indices)
    pub adjacency_indices: Vec<usize>,
    /// Global simulation time
    pub time: f64,
}

impl Network {
    pub fn new(size: usize) -> Self {
        Self {
            name: "NeuralSim Network".into(),
            neurons: NeuronArray::new(size),
            adjacency_ptr: vec![0; size + 1],
            synapses: Vec::new(),
            adjacency_indices: Vec::new(),
            time: 0.0,
        }
    }

    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Add a single synapse from source to target
    pub fn add_synapse(&mut self, synapse: SynapseState) {
        let source = synapse.source;
        // Shift all pointers for neurons after source
        for ptr in self.adjacency_ptr.iter_mut().skip(source + 1) {
            *ptr += 1;
        }
        self.adjacency_indices.push(synapse.target);
        self.synapses.push(synapse);
    }

    pub fn connect_random(&mut self, probability: f64, rng: &mut impl rand::Rng) {
        use crate::synapse::{SynapseState, SynapseType};
        let n = self.neuron_count();
        for i in 0..n {
            for j in 0..n {
                if i != j && rng.random::<f64>() < probability {
                    let syn_type = match self.neurons.neuron_type[i] {
                        NeuronType::Excitatory => SynapseType::AMPA,
                        NeuronType::Inhibitory => SynapseType::GABA,
                    };
                    self.add_synapse(SynapseState::new(i, j, 1.0, syn_type));
                }
            }
        }
    }
}

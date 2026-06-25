pub mod graph;
pub mod region;
pub mod connectivity;
pub mod builder;

use crate::neuron::{NeuronArray, NeuronType};
use crate::synapse::{SynapseState, SynapseType};
use serde::{Deserialize, Serialize};

pub type RegionId = usize;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub name: String,
    pub neurons: NeuronArray,
    pub adjacency_ptr: Vec<usize>,
    pub synapses: Vec<SynapseState>,
    pub adjacency_indices: Vec<usize>,
    pub time: f64,
    /// Maps neuron index -> region ID
    pub neuron_region: Vec<RegionId>,
    /// Region names for display/debug
    pub region_names: Vec<String>,
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
            neuron_region: vec![0; size],
            region_names: vec!["Default".into()],
        }
    }

    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    pub fn add_synapse(&mut self, synapse: SynapseState) {
        let source = synapse.source;
        for ptr in self.adjacency_ptr.iter_mut().skip(source + 1) {
            *ptr += 1;
        }
        self.adjacency_indices.push(synapse.target);
        self.synapses.push(synapse);
    }

    pub fn connect_random(&mut self, probability: f64, rng: &mut impl rand::Rng) {
        let n = self.neuron_count();
        for i in 0..n {
            for j in 0..n {
                if i != j && rng.random::<f64>() < probability {
                    let syn_type = match self.neurons.neuron_type[i] {
                        NeuronType::Excitatory => SynapseType::AMPA,
                        NeuronType::Inhibitory => SynapseType::GABA,
                    };
                    let weight = if syn_type == SynapseType::GABA { -rng.random::<f64>() * 2.0 } else { rng.random::<f64>() * 3.0 + 1.0 };
                    self.add_synapse(SynapseState::new(i, j, weight, syn_type));
                }
            }
        }
    }

    /// Count synapses per region
    pub fn region_counts(&self) -> Vec<(String, usize, usize)> {
        let mut neuron_counts: Vec<usize> = vec![0; self.region_names.len()];
        let mut spike_counts: Vec<u64> = vec![0; self.region_names.len()];
        for i in 0..self.neuron_count() {
            let rid = self.neuron_region[i];
            if rid < neuron_counts.len() {
                neuron_counts[rid] += 1;
                spike_counts[rid] += self.neurons.spike_count[i];
            }
        }
        self.region_names.iter().enumerate().map(|(i, name)| {
            (name.clone(), neuron_counts[i], spike_counts[i] as usize)
        }).collect()
    }

    /// Compute mean firing rate per region
    pub fn region_rates(&self) -> Vec<(String, f64)> {
        let mut neuron_counts: Vec<usize> = vec![0; self.region_names.len()];
        let mut spike_counts: Vec<u64> = vec![0; self.region_names.len()];
        for i in 0..self.neuron_count() {
            let rid = self.neuron_region[i];
            if rid < neuron_counts.len() {
                neuron_counts[rid] += 1;
                spike_counts[rid] += self.neurons.spike_count[i];
            }
        }
        let t = self.time.max(0.001);
        self.region_names.iter().enumerate().map(|(i, name)| {
            let rate = if neuron_counts[i] > 0 {
                spike_counts[i] as f64 / neuron_counts[i] as f64 / (t / 1000.0)
            } else { 0.0 };
            (name.clone(), rate)
        }).collect()
    }
}

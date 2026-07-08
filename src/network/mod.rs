pub mod graph;
pub mod region;
pub mod connectivity;
pub mod builder;
pub mod simple_builder;

pub use simple_builder::NetworkBuilder;

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

    /// Add a synapse (deferred adjacency update — call finalize() after all adds).
    pub fn add_synapse(&mut self, synapse: SynapseState) {
        self.synapses.push(synapse);
    }

    /// Build CSR adjacency from synapses in O(N + M) time.
    /// Must be called once after all synapses have been added via add_synapse().
    pub fn finalize(&mut self) {
        let n = self.neuron_count();
        let m = self.synapses.len();
        let mut degree = vec![0u32; n];
        for syn in &self.synapses {
            if syn.source < n {
                degree[syn.source] += 1;
            }
        }
        let mut ptr = vec![0usize; n + 1];
        let mut sum = 0usize;
        for (d, p) in degree.iter().zip(ptr.iter_mut().skip(1)) {
            sum += *d as usize;
            *p = sum;
        }
        let mut indices = vec![0usize; m];
        // Place each target into its source's bucket using a write cursor per source
        let mut cursor: Vec<usize> = ptr[..n].to_vec();
        for syn in &self.synapses {
            if syn.source < n {
                let pos = &mut cursor[syn.source];
                indices[*pos] = syn.target;
                *pos += 1;
            }
        }
        self.adjacency_ptr = ptr;
        self.adjacency_indices = indices;
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
        self.finalize();
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

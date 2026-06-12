use crate::neuron::lif::LifNeuron;
use crate::neuron::{NeuronModel, NeuronState};
use crate::network::Network;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationStats {
    pub total_spikes: u64,
    pub active_neurons: u64,
    pub mean_firing_rate: f64,
    pub sim_time_ms: f64,
}

pub struct SimulationEngine {
    pub network: Arc<RwLock<Network>>,
    pub dt: f64,
    pub spike_buffer: Vec<(usize, f64)>,
    pub stats: Arc<RwLock<SimulationStats>>,
}

impl SimulationEngine {
    pub fn new(network: Network) -> Self {
        Self {
            network: Arc::new(RwLock::new(network)),
            dt: 0.1,
            spike_buffer: Vec::with_capacity(1024),
            stats: Arc::new(RwLock::new(SimulationStats::default())),
        }
    }

    pub fn step(&mut self) {
        let spike_count = Arc::new(AtomicU64::new(0));
        let dt = self.dt;

        // Phase 1: Extract neuron state, update in parallel, write back
        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            // Extract current states into local vecs for parallel processing
            let mut states: Vec<NeuronState> = (0..n)
                .map(|i| {
                    NeuronState {
                        membrane_potential: net.neurons.membrane_potential[i],
                        recovery_variable: net.neurons.recovery_variable[i],
                        refractory_counter: net.neurons.refractory_counter[i],
                        last_spike_time: net.neurons.last_spike_time[i],
                        spike_count: net.neurons.spike_count[i],
                        neuron_type: net.neurons.neuron_type[i],
                        model_params: net.neurons.model_params[i],
                    }
                })
                .collect();

            // Update states in parallel
            let input_currents: Vec<f64> = net.neurons.input_current.clone();
            let spikes: Vec<bool> = states
                .par_iter_mut()
                .enumerate()
                .map(|(i, state)| {
                    let mut lif = LifNeuron;
                    let spiked = lif.step(state, dt, input_currents[i]);
                    if spiked {
                        spike_count.fetch_add(1, Ordering::Relaxed);
                    }
                    spiked
                })
                .collect();

            // Write back
            for (i, state) in states.into_iter().enumerate() {
                net.neurons.membrane_potential[i] = state.membrane_potential;
                net.neurons.recovery_variable[i] = state.recovery_variable;
                net.neurons.refractory_counter[i] = state.refractory_counter;
                net.neurons.last_spike_time[i] = state.last_spike_time;
                net.neurons.spike_count[i] = state.spike_count;
            }

            // Collect spikes
            self.spike_buffer.clear();
            for (i, spiked) in spikes.iter().enumerate() {
                if *spiked {
                    self.spike_buffer.push((i, net.time));
                }
            }
        }

        // Phase 2: Propagate spikes through synapses
        {
            let mut net = self.network.write();
            for &(neuron_id, _time) in self.spike_buffer.iter() {
                let start = net.adjacency_ptr[neuron_id];
                let end = net.adjacency_ptr[neuron_id + 1];
                for idx in start..end {
                    let target = net.adjacency_indices[idx];
                    if target < net.neurons.len() {
                        let weight = net.synapses.get(idx).map(|s| s.weight).unwrap_or(0.0);
                        net.neurons.input_current[target] += weight;
                    }
                }
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            let spiked = spike_count.load(Ordering::Relaxed);
            stats.total_spikes += spiked;
            stats.active_neurons = stats.active_neurons.max(spiked);
        }

        // Advance time and decay input currents
        {
            let mut net = self.network.write();
            net.time += dt;
            for c in net.neurons.input_current.iter_mut() {
                *c *= 0.9;
            }
        }
    }

    pub fn simulate_ms(&mut self, duration_ms: f64) {
        let steps = (duration_ms / self.dt) as usize;
        for _ in 0..steps {
            self.step();
        }
    }

    pub fn stats(&self) -> SimulationStats {
        self.stats.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_step() {
        let network = Network::new(100);
        let mut engine = SimulationEngine::new(network);
        engine.simulate_ms(10.0);
        let stats = engine.stats();
        assert!(stats.sim_time_ms >= 0.0);
    }
}

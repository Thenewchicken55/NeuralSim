use crate::neuron::lif::LifNeuron;
use crate::neuron::{NeuronModel, NeuronState};
use crate::network::Network;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimulationStats {
    pub total_spikes: u64,
    pub output_spikes: u64,
    pub active_neurons: u64,
    pub sim_time_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct StepResult {
    pub spike_count: u64,
    pub output_spike_count: u64,
    pub spiking_neurons: Vec<usize>,
    pub output_spiking_neurons: Vec<usize>,
}

pub struct SimulationEngine {
    pub network: Arc<RwLock<Network>>,
    pub dt: f64,
    pub spike_buffer: Vec<(usize, f64)>,
    pub stats: Arc<RwLock<SimulationStats>>,
    rng: StdRng,
    pub noise_amplitude: f64,
    noise_density: f64,
    external_stimulus_chance: f64,
    external_stimulus_strength: f64,
}

impl SimulationEngine {
    pub fn new(network: Network) -> Self {
        Self {
            network: Arc::new(RwLock::new(network)),
            dt: 0.5,
            spike_buffer: Vec::with_capacity(4096),
            stats: Arc::new(RwLock::new(SimulationStats::default())),
            rng: StdRng::seed_from_u64(42),
            noise_amplitude: 8.0,
            noise_density: 0.3,
            external_stimulus_chance: 0.01,
            external_stimulus_strength: 20.0,
        }
    }

    pub fn with_noise(mut self, amplitude: f64) -> Self {
        self.noise_amplitude = amplitude;
        self
    }

    /// Inject current into a specific neuron (call from GUI)
    pub fn stimulate(&mut self, neuron_id: usize, current: f64) {
        let mut net = self.network.write();
        if neuron_id < net.neurons.len() {
            net.neurons.input_current[neuron_id] += current;
        }
    }

    pub fn step(&mut self) -> StepResult {
        let spike_count = Arc::new(AtomicU64::new(0));
        let output_spike_count = Arc::new(AtomicU64::new(0));
        let dt = self.dt;

        let mut local_spiking = Vec::new();
        let mut local_output_spiking = Vec::new();

        // Phase 1: Update neurons
        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            // Inject background noise
            for i in 0..n {
                if self.rng.random::<f64>() < self.noise_density {
                    net.neurons.input_current[i] += self.rng.random::<f64>() * self.noise_amplitude;
                }
            }

            // Random external stimulus events (like sensory input)
            if self.rng.random::<f64>() < self.external_stimulus_chance {
                let count = self.rng.random_range(5..=30);
                for _ in 0..count {
                    let target = self.rng.random_range(0..n);
                    net.neurons.input_current[target] += self.rng.random::<f64>() * self.external_stimulus_strength;
                }
            }

            let mut states: Vec<NeuronState> = (0..n)
                .map(|i| NeuronState {
                    membrane_potential: net.neurons.membrane_potential[i],
                    recovery_variable: net.neurons.recovery_variable[i],
                    refractory_counter: net.neurons.refractory_counter[i],
                    last_spike_time: net.neurons.last_spike_time[i],
                    spike_count: net.neurons.spike_count[i],
                    neuron_type: net.neurons.neuron_type[i],
                    model_params: net.neurons.model_params[i],
                })
                .collect();

            let input_currents: Vec<f64> = net.neurons.input_current.clone();
            let is_output: Vec<bool> = net.neurons.is_output.clone();
            let spikes: Vec<bool> = states
                .par_iter_mut()
                .enumerate()
                .map(|(i, state)| {
                    let mut lif = LifNeuron;
                    let spiked = lif.step(state, dt, input_currents[i]);
                    if spiked {
                        spike_count.fetch_add(1, Ordering::Relaxed);
                        if is_output[i] {
                            output_spike_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    spiked
                })
                .collect();

            for (i, state) in states.into_iter().enumerate() {
                net.neurons.membrane_potential[i] = state.membrane_potential;
                net.neurons.recovery_variable[i] = state.recovery_variable;
                net.neurons.refractory_counter[i] = state.refractory_counter;
                net.neurons.last_spike_time[i] = state.last_spike_time;
                net.neurons.spike_count[i] = state.spike_count;
            }

            self.spike_buffer.clear();
            for (i, spiked) in spikes.iter().enumerate() {
                if *spiked {
                    self.spike_buffer.push((i, net.time));
                    local_spiking.push(i);
                    if is_output[i] {
                        local_output_spiking.push(i);
                    }
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

        // Phase 3: Advance time and decay currents
        let sim_time;
        {
            let mut net = self.network.write();
            net.time += dt;
            sim_time = net.time;
            // Slower decay so activity reverberates longer
            for c in net.neurons.input_current.iter_mut() {
                *c *= 0.92;
            }
        }

        // Phase 4: Stats
        {
            let mut stats = self.stats.write();
            let spiked = spike_count.load(Ordering::Relaxed);
            let output = output_spike_count.load(Ordering::Relaxed);
            stats.total_spikes += spiked;
            stats.output_spikes += output;
            stats.active_neurons = stats.active_neurons.max(spiked);
            stats.sim_time_ms = sim_time;
        }

        StepResult {
            spike_count: spike_count.load(Ordering::Relaxed),
            output_spike_count: output_spike_count.load(Ordering::Relaxed),
            spiking_neurons: local_spiking,
            output_spiking_neurons: local_output_spiking,
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
        let mut network = Network::new(100);
        let mut rng = StdRng::seed_from_u64(42);
        network.connect_random(0.05, &mut rng);
        let mut engine = SimulationEngine::new(network);
        engine.simulate_ms(50.0);
        let stats = engine.stats();
        assert!(stats.sim_time_ms >= 0.0);
    }

    #[test]
    fn test_engine_produces_spikes() {
        let mut network = Network::new(200);
        let mut rng = StdRng::seed_from_u64(42);
        network.connect_random(0.05, &mut rng);
        let mut engine = SimulationEngine::new(network).with_noise(10.0);
        engine.simulate_ms(100.0);
        let stats = engine.stats();
        assert!(stats.total_spikes > 0, "Engine should produce spikes with noise");
    }
}

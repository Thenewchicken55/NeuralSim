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
    pub active_neurons: u64,
    pub mean_firing_rate: f64,
    pub sim_time_ms: f64,
}

pub struct SimulationEngine {
    pub network: Arc<RwLock<Network>>,
    pub dt: f64,
    pub spike_buffer: Vec<(usize, f64)>,
    pub stats: Arc<RwLock<SimulationStats>>,
    rng: StdRng,
    noise_amplitude: f64,
}

impl SimulationEngine {
    pub fn new(network: Network) -> Self {
        Self {
            network: Arc::new(RwLock::new(network)),
            dt: 0.5,
            spike_buffer: Vec::with_capacity(1024),
            stats: Arc::new(RwLock::new(SimulationStats::default())),
            rng: StdRng::seed_from_u64(42),
            noise_amplitude: 5.0,
        }
    }

    pub fn with_noise(mut self, amplitude: f64) -> Self {
        self.noise_amplitude = amplitude;
        self
    }

    pub fn step(&mut self) {
        let spike_count = Arc::new(AtomicU64::new(0));
        let dt = self.dt;
        let noise_amp = self.noise_amplitude;

        // Phase 1: Update all neurons in parallel
        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            // Inject background noise to random neurons
            for i in 0..n {
                if self.rng.random::<f64>() < 0.2 {
                    net.neurons.input_current[i] += self.rng.random::<f64>() * noise_amp;
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
                }
            }
        }

        // Phase 2: Propagate spikes
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
            for c in net.neurons.input_current.iter_mut() {
                *c *= 0.8;
            }
        }

        // Phase 4: Update stats
        {
            let mut stats = self.stats.write();
            let spiked = spike_count.load(Ordering::Relaxed);
            stats.total_spikes += spiked;
            stats.active_neurons = stats.active_neurons.max(spiked);
            stats.sim_time_ms = sim_time;
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

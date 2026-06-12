use crate::neuron::lif::LifNeuron;
use crate::neuron::{NeuronModel, NeuronState};
#[cfg(feature = "gpu")]
use crate::neuron::NeuronModelParams;
use crate::network::Network;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use crate::simulation::gpu_backend::GpuBackend;

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
    #[cfg(feature = "gpu")]
    pub gpu_backend: Option<GpuBackend>,
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
            #[cfg(feature = "gpu")]
            gpu_backend: None,
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

    /// Try to initialise the GPU compute backend.
    ///
    /// Uploads all current neuron state to GPU buffers.  If this fails
    /// (e.g. no GPU available), the engine falls back to CPU automatically.
    #[cfg(feature = "gpu")]
    pub fn init_gpu(&mut self) -> Result<(), crate::simulation::gpu_backend::GpuError> {
        let (num_neurons, num_edges) = {
            let net = self.network.read();
            (net.neuron_count(), net.synapse_count())
        };
        let mut gpu = GpuBackend::new(num_neurons, num_edges)?;

        {
            let net = self.network.read();
            let n = net.neuron_count();
            let mem_pot: Vec<f32> = net.neurons.membrane_potential.iter().map(|&v| v as f32).collect();
            let rec_var: Vec<f32> = net.neurons.recovery_variable.iter().map(|&v| v as f32).collect();
            let refr_ctr: Vec<i32> = net.neurons.refractory_counter.clone();
            let last_time: Vec<f32> = net.neurons.last_spike_time.iter().map(|&v| v as f32).collect();
            let spike_ct: Vec<u32> = net.neurons.spike_count.iter().map(|&v| v as u32).collect();
            let is_output: Vec<u32> = net.neurons.is_output.iter().map(|&b| if b { 1 } else { 0 }).collect();

            let params: Vec<[f32; 6]> = (0..n)
                .map(|i| match &net.neurons.model_params[i] {
                    NeuronModelParams::Lif {
                        resting, threshold, reset, tau_m, refractory_period, input_resistance,
                    } => [
                        *resting as f32,
                        *threshold as f32,
                        *reset as f32,
                        *tau_m as f32,
                        *refractory_period as f32,
                        *input_resistance as f32,
                    ],
                    _ => [0.0; 6],
                })
                .collect();

            gpu.upload_initial_state(&mem_pot, &rec_var, &refr_ctr, &last_time, &spike_ct, &params, &is_output);

            // Upload CSR adjacency data
            let adj_ptr: Vec<u32> = net.adjacency_ptr.iter().map(|&v| v as u32).collect();
            let adj_indices: Vec<u32> = net.adjacency_indices.iter().map(|&v| v as u32).collect();
            let weights: Vec<f32> = net.synapses.iter().map(|s| s.weight as f32).collect();
            gpu.upload_csr(&adj_ptr, &adj_indices, &weights);
        }

        self.gpu_backend = Some(gpu);
        Ok(())
    }

    pub fn step(&mut self) -> StepResult {
        // Use GPU backend when available
        #[cfg(feature = "gpu")]
        if self.gpu_backend.is_some() {
            return self.step_gpu();
        }
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

    /// Step the simulation using the GPU compute backend.
    #[cfg(feature = "gpu")]
    fn step_gpu(&mut self) -> StepResult {
        let dt = self.dt;
        let total_spike_count = Arc::new(AtomicU64::new(0));
        let total_output_spike_count = Arc::new(AtomicU64::new(0));
        let mut local_spiking = Vec::new();
        let mut local_output_spiking = Vec::new();

        let input_f32: Vec<f32>;
        let sim_time: f64;

        // Phase 1: noise injection & external stimulus (CPU), then GPU LIF update
        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            for i in 0..n {
                if self.rng.random::<f64>() < self.noise_density {
                    net.neurons.input_current[i] += self.rng.random::<f64>() * self.noise_amplitude;
                }
            }
            if self.rng.random::<f64>() < self.external_stimulus_chance {
                let count = self.rng.random_range(5..=30);
                for _ in 0..count {
                    let target = self.rng.random_range(0..n);
                    net.neurons.input_current[target] += self.rng.random::<f64>() * self.external_stimulus_strength;
                }
            }

            input_f32 = net.neurons.input_current.iter().map(|&v| v as f32).collect();
        }

        // GPU LIF step
        let gpu = self.gpu_backend.as_mut().unwrap();
        gpu.upload_input_current(&input_f32);
        gpu.write_uniforms(dt as f32, self.network.read().time as f32);
        let (spiked, mem_pot, rec_var, refr_ctr, spike_ct) = gpu.step_lif();

        // GPU SpMV (synapse propagation on GPU) — returns updated input_current
        let gpu_input_current = gpu.step_spmv();

        // Phase 2: update CPU state, collect spikes, decay & advance time
        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            for i in 0..n {
                net.neurons.membrane_potential[i] = mem_pot[i] as f64;
                net.neurons.recovery_variable[i] = rec_var[i] as f64;
                net.neurons.refractory_counter[i] = refr_ctr[i];
                net.neurons.spike_count[i] = spike_ct[i] as u64;
                // Sync CPU-side input_current with GPU (includes SpMV contributions)
                net.neurons.input_current[i] = gpu_input_current[i] as f64;
            }

            self.spike_buffer.clear();
            for (i, &s) in spiked.iter().enumerate() {
                if s != 0 {
                    total_spike_count.fetch_add(1, Ordering::Relaxed);
                    if net.neurons.is_output[i] {
                        total_output_spike_count.fetch_add(1, Ordering::Relaxed);
                        local_output_spiking.push(i);
                    }
                    self.spike_buffer.push((i, net.time));
                    local_spiking.push(i);
                }
            }

            net.time += dt;
            sim_time = net.time;
            for c in net.neurons.input_current.iter_mut() {
                *c *= 0.92;
            }
        }

        // Stats
        {
            let mut stats = self.stats.write();
            let sp = total_spike_count.load(Ordering::Relaxed);
            let op = total_output_spike_count.load(Ordering::Relaxed);
            stats.total_spikes += sp;
            stats.output_spikes += op;
            stats.active_neurons = stats.active_neurons.max(sp);
            stats.sim_time_ms = sim_time;
        }

        StepResult {
            spike_count: total_spike_count.load(Ordering::Relaxed),
            output_spike_count: total_output_spike_count.load(Ordering::Relaxed),
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

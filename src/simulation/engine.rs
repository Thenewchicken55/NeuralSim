use crate::neuron::lif::LifNeuron;
use crate::neuron::izhikevich::IzhikevichNeuron;
use crate::neuron::hodgkin_huxley::HodgkinHuxleyNeuron;
use crate::neuron::{NeuronModel, NeuronModelParams, NeuronState};
use crate::network::Network;
use crate::synapse::{PlasticityConfig, SynapseType};

use crate::synapse::types::dynamics_for;
use parking_lot::RwLock;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    pub weight_updates: u64,
    pub mean_firing_rate: f64,
    pub synchrony_index: f64,
    pub weight_mean: f64,
    pub weight_std: f64,
    pub seed: u64,
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
    /// Global plasticity configuration
    pub plasticity: PlasticityConfig,
    /// Per-neuron firing rate estimates (for homeostatic / BCM)
    firing_rates: Vec<f64>,
    /// Per-neuron BCM theta thresholds
    _bcm_theta: Vec<f64>,
    /// Step counter for periodic tasks
    step_count: u64,
    /// Synaptic conductance buffers (per synapse, for alpha function)
    conductances: Vec<f64>,
    /// Whether to use conductance-based dynamics
    pub use_conductance: bool,
    /// LFP signal (sum of absolute synaptic currents)
    pub lfp_signal: f64,
    /// Spike history for synchrony (last N steps)
    spike_history: Vec<Vec<bool>>,
    synchrony_window: usize,
    /// Pre-built index: for each target neuron, list of incoming synapse indices
    synapses_by_target: Vec<Vec<usize>>,
    /// Reward signal for R-STDP (dopamine-like modulation)
    pub reward_signal: f64,
    /// Number of output spikes in the current step (used to compute reward)
    reward_output_spike_count: u64,
    /// If output spikes exceed this threshold, reward is applied
    pub reward_threshold: u64,
    /// Reward decay per step (exponential decay toward 0)
    pub reward_decay: f64,
    #[cfg(feature = "gpu")]
    pub gpu_backend: Option<GpuBackend>,
}

impl SimulationEngine {
    pub fn new(network: Network) -> Self {
        let n = network.neuron_count();
        let n_syn = network.synapse_count();
        let mut synapses_by_target = vec![Vec::new(); n];
        for idx in 0..n_syn {
            let target = network.synapses[idx].target;
            if target < n {
                synapses_by_target[target].push(idx);
            }
        }
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
            plasticity: PlasticityConfig::default(),
            firing_rates: vec![0.0; n],
            _bcm_theta: vec![0.5; n],
            step_count: 0,
            conductances: vec![0.0; n_syn],
            use_conductance: true,
            lfp_signal: 0.0,
            spike_history: Vec::new(),
            synchrony_window: 100,
            synapses_by_target,
            reward_signal: 0.0,
            reward_output_spike_count: 0,
            reward_threshold: 5,
            reward_decay: 0.95,
            #[cfg(feature = "gpu")]
            gpu_backend: None,
        }
    }

    /// Set the RNG seed used for this simulation.
    /// This is stored in stats for checkpoint metadata.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self.stats.write().seed = seed;
        self
    }

    pub fn with_plasticity(mut self, config: PlasticityConfig) -> Self {
        self.plasticity = config;
        self
    }

    pub fn with_noise(mut self, amplitude: f64) -> Self {
        self.noise_amplitude = amplitude;
        self
    }

    pub fn with_conductance(mut self, enable: bool) -> Self {
        self.use_conductance = enable;
        self
    }

    pub fn with_reward(mut self, threshold: u64, decay: f64) -> Self {
        self.reward_threshold = threshold;
        self.reward_decay = decay;
        self
    }

    pub fn with_reward_signal(mut self, signal: f64) -> Self {
        self.reward_signal = signal;
        self
    }

    pub fn stimulate(&mut self, neuron_id: usize, current: f64) {
        let mut net = self.network.write();
        if neuron_id < net.neurons.len() {
            net.neurons.input_current[neuron_id] += current;
        }
    }

    /// Apply a dopamine-like reward signal for R-STDP
    pub fn apply_reward(&mut self, reward: f64) {
        let mut net = self.network.write();
        if let Some(rstdp) = &self.plasticity.rstdp {
            for syn in net.synapses.iter_mut() {
                if let Some(ref mut et) = syn.eligibility {
                    et.apply_reward(reward, &mut syn.weight, rstdp.learning_rate,
                                    rstdp.weight_min, rstdp.weight_max);
                }
            }
        }
    }

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
                    NeuronModelParams::Lif { resting, threshold, reset, tau_m, refractory_period, input_resistance } => [
                        *resting as f32, *threshold as f32, *reset as f32,
                        *tau_m as f32, *refractory_period as f32, *input_resistance as f32,
                    ],
                    _ => [0.0; 6],
                })
                .collect();

            gpu.upload_initial_state(&mem_pot, &rec_var, &refr_ctr, &last_time, &spike_ct, &params, &is_output);

            let adj_ptr: Vec<u32> = net.adjacency_ptr.iter().map(|&v| v as u32).collect();
            let adj_indices: Vec<u32> = net.adjacency_indices.iter().map(|&v| v as u32).collect();
            let weights: Vec<f32> = net.synapses.iter().map(|s| s.weight as f32).collect();
            gpu.upload_csr(&adj_ptr, &adj_indices, &weights);
        }

        self.gpu_backend = Some(gpu);
        Ok(())
    }

    pub fn step(&mut self) -> StepResult {
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
        let (_all_spiked, sim_time) = {
            let mut net = self.network.write();
            let n = net.neuron_count();

            // Inject background noise
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

            // Operate directly on SoA fields via index-based parallelism
            // to avoid per-step Vec<NeuronState> allocation and write-back.
            let idxs: Vec<usize> = (0..n).collect();
            let input_currents: Vec<f64> = net.neurons.input_current.clone();
            let is_output: Vec<bool> = net.neurons.is_output.clone();
            let model_params: Vec<NeuronModelParams> = net.neurons.model_params.clone();

            // We need to mutate the SoA fields in parallel. Split into slices.
            let mem_pot = &mut net.neurons.membrane_potential;
            let rec_var = &mut net.neurons.recovery_variable;
            let refr_ctr = &mut net.neurons.refractory_counter;
            let last_time = &mut net.neurons.last_spike_time;
            let spike_ct = &mut net.neurons.spike_count;
            let hh_m = &mut net.neurons.hh_m;
            let hh_h = &mut net.neurons.hh_h;
            let hh_n = &mut net.neurons.hh_n;
            let just_spiked = &mut net.neurons.just_spiked;

            let spikes: Vec<bool> = idxs
                .par_iter()
                .map(|&i| {
                    let mut state = NeuronState {
                        membrane_potential: mem_pot[i],
                        recovery_variable: rec_var[i],
                        refractory_counter: refr_ctr[i],
                        last_spike_time: last_time[i],
                        spike_count: spike_ct[i],
                        neuron_type: net.neurons.neuron_type[i],
                        model_params: model_params[i],
                        hh_m: hh_m[i],
                        hh_h: hh_h[i],
                        hh_n: hh_n[i],
                        just_spiked: just_spiked[i],
                    };
                    let spiked = Self::step_neuron(i, &mut state, dt, input_currents[i], &model_params);
                    if spiked {
                        spike_count.fetch_add(1, Ordering::Relaxed);
                        if is_output[i] {
                            output_spike_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    // Write back directly to SoA
                    mem_pot[i] = state.membrane_potential;
                    rec_var[i] = state.recovery_variable;
                    refr_ctr[i] = state.refractory_counter;
                    last_time[i] = state.last_spike_time;
                    spike_ct[i] = state.spike_count;
                    hh_m[i] = state.hh_m;
                    hh_h[i] = state.hh_h;
                    hh_n[i] = state.hh_n;
                    just_spiked[i] = state.just_spiked;
                    spiked
                })
                .collect();

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

            // Track spike history for synchrony
            if self.synchrony_window > 0 {
                self.spike_history.push(spikes.clone());
                if self.spike_history.len() > self.synchrony_window {
                    self.spike_history.remove(0);
                }
            }

            let st = net.time;
            net.time += dt;
            (spikes, st)
        };

        // Build spike_set for O(1) STDP lookups
        let spike_set: HashSet<usize> = self.spike_buffer.iter().map(|&(nid, _)| nid).collect();

        // Phase 2: Propagate spikes through synapses with STDP and conductance
        {
            let mut net = self.network.write();
            let n_syn = net.synapses.len();

            // Ensure conductance buffer is sized correctly
            if self.conductances.len() != n_syn {
                self.conductances.resize(n_syn, 0.0);
            }

            // Decay all conductances
            for (idx, syn) in net.synapses.iter().enumerate() {
                let dyns = dynamics_for(&syn.synapse_type);
                if self.use_conductance && idx < self.conductances.len() {
                    self.conductances[idx] *= f64::exp(-dt / dyns.decay_time);
                }
            }

            // Process spikes (track per-synapse contribution)
            let contribution_decay = f64::exp(-dt / self.plasticity.contribution_tau);
            for &(neuron_id, spike_time) in self.spike_buffer.iter() {
                let start = net.adjacency_ptr[neuron_id];
                let end = net.adjacency_ptr[neuron_id + 1];
                for idx in start..end {
                    if idx >= n_syn { break; }
                    let target = net.adjacency_indices[idx];
                    if target >= net.neurons.len() { continue; }

                    let syn = &net.synapses[idx];
                    let delay = syn.delay;
                    let _delivery_time = spike_time + delay;

                    if self.use_conductance {
                        // Alpha-function conductance
                        let dyns = dynamics_for(&syn.synapse_type);
                        if idx < self.conductances.len() {
                            // NMDA voltage gating
                            let v = net.neurons.membrane_potential[target];
                            let mg_factor = dyns.nmda_mg_block(v);
                            // Peak conductance at t=0 after spike
                            let g_peak = syn.weight * dyns.conductance_max * mg_factor;

                            if syn.synapse_type == SynapseType::GABA || syn.synapse_type == SynapseType::GabaB {
                                self.conductances[idx] += g_peak.abs();
                            } else {
                                self.conductances[idx] += g_peak;
                            }
                            // Track per-synapse contribution (conductance magnitude)
                            if idx < n_syn {
                                net.synapses[idx].contribution_avg =
                                    net.synapses[idx].contribution_avg * contribution_decay
                                    + g_peak.abs() * (1.0 - contribution_decay);
                            }
                        }
                    } else {
                        // Direct current injection (simplified)
                        let mut current = syn.weight;
                        if let Some(ref stp) = syn.stp {
                            // Apply short-term plasticity
                            let mut stp_local = *stp;
                            current *= stp_local.step(dt, true);
                        }
                        net.neurons.input_current[target] += current;
                        // Track per-synapse contribution (current magnitude)
                        if idx < n_syn {
                            net.synapses[idx].contribution_avg =
                                net.synapses[idx].contribution_avg * contribution_decay
                                + current.abs() * (1.0 - contribution_decay);
                        }
                    }
                }
            }

            // Apply conductance-based currents
            if self.use_conductance {
                for idx in 0..n_syn {
                    let syn = &net.synapses[idx];
                    let g = self.conductances[idx];
                    if g.abs() < 1e-12 { continue; }
                    let target = syn.target;
                    if target >= net.neurons.len() { continue; }
                    let dyns = dynamics_for(&syn.synapse_type);
                    let v = net.neurons.membrane_potential[target];
                    // I_syn = g(t) * (V - E_rev)
                    let i_syn = g * (dyns.reversal_potential - v);
                    net.neurons.input_current[target] += i_syn;
                    self.lfp_signal += i_syn.abs();
                }
            }

            // STDP weight updates
            if self.plasticity.enabled {
                let current_time = net.time;
                if let Some(ref stdp) = self.plasticity.stdp {
                    for syn in net.synapses.iter_mut() {
                        if !syn.plasticity_enabled || syn.stdp_trace.is_none() { continue; }
                        if let Some(ref mut trace) = syn.stdp_trace {
                            // Decay traces
                            trace.decay(dt, stdp.tau_plus, stdp.tau_minus);

                            // Check if pre/post spike occurred during this step (O(1) via HashSet)
                            let pre_fired = spike_set.contains(&syn.source);
                            let post_fired = spike_set.contains(&syn.target);

                            if pre_fired {
                                // LTP: pre before post — weight increases based on post-trace
                                let delta = stdp.a_plus * trace.post_trace;
                                syn.weight = (syn.weight + delta).clamp(stdp.weight_min, stdp.weight_max);
                                trace.on_pre_spike(current_time, stdp.tau_plus);

                                // Update eligibility trace for R-STDP
                                if let Some(ref mut et) = syn.eligibility {
                                    et.update(dt, 200.0, delta);
                                }
                            }
                            if post_fired {
                                // LTD: post before pre — weight decreases based on pre-trace
                                let delta = -stdp.a_minus * trace.pre_trace;
                                syn.weight = (syn.weight + delta).clamp(stdp.weight_min, stdp.weight_max);
                                trace.on_post_spike(current_time, stdp.tau_minus);

                                if let Some(ref mut et) = syn.eligibility {
                                    et.update(dt, 200.0, delta);
                                }
                            }
                        }
                    }
                }

                // Consolidation
                if let Some(ref consol) = self.plasticity.consolidation {
                    for syn in net.synapses.iter_mut() {
                        consol.apply(&mut syn.weight, dt);
                    }
                }

                // R-STDP: apply reward signal if output spikes exceed threshold
                if self.plasticity.rstdp.is_some() && self.reward_signal.abs() > 0.001 {
                    for syn in net.synapses.iter_mut() {
                        if let Some(ref mut et) = syn.eligibility {
                            if let Some(ref rstdp) = self.plasticity.rstdp {
                                et.apply_reward(
                                    self.reward_signal,
                                    &mut syn.weight,
                                    rstdp.learning_rate,
                                    rstdp.weight_min,
                                    rstdp.weight_max,
                                );
                            }
                        }
                    }
                }
                // Decay reward signal
                self.reward_signal *= self.reward_decay;

                // Per-synapse homeostatic scaling
                // Each synapse is scaled toward a target contribution level.
                // If postsynaptic rate exceeds target, weights decrease;
                // if rate is below target, weights increase.
                if self.plasticity.homeostatic_tau > 0.0 {
                    let target_rate = self.plasticity.homeostatic_target_rate;
                    for i in 0..net.neuron_count() {
                        let rate = self.firing_rates[i];
                        let error = rate - target_rate;

                        if let Some(syn_indices) = self.synapses_by_target.get(i) {
                            // Compute mean contribution across synapses targeting this neuron
                            let mut sum_contrib = 0.0;
                            let mut count = 0usize;
                            for &idx in syn_indices {
                                if idx < n_syn {
                                    sum_contrib += net.synapses[idx].contribution_avg;
                                    count += 1;
                                }
                            }
                            if count == 0 || sum_contrib < 1e-12 { continue; }
                            let mean_contrib = sum_contrib / count as f64;

                            for &idx in syn_indices {
                                if idx >= n_syn { continue; }
                                let syn = &mut net.synapses[idx];
                                // Over-contributing synapses feel more error, under-contributing less
                                let rel_contrib = syn.contribution_avg / mean_contrib;
                                let per_syn_error = error * rel_contrib;
                                // Scale < 1.0 when over-firing (reduces weight)
                                let scale = f64::exp(-per_syn_error * dt / self.plasticity.homeostatic_tau);
                                syn.weight *= scale;
                                syn.weight = syn.weight.clamp(0.0, 10.0);
                            }
                        }
                    }
                }
            }

            // Decay synaptic currents (for direct mode, not conductance mode)
            if !self.use_conductance {
                for c in net.neurons.input_current.iter_mut() {
                    *c *= 0.92;
                }
            }
        }

        // Phase 3: Update firing rate estimates and stats
        {
            let mut stats = self.stats.write();
            let spiked = spike_count.load(Ordering::Relaxed);
            let output = output_spike_count.load(Ordering::Relaxed);
            stats.total_spikes += spiked;
            stats.output_spikes += output;
            stats.active_neurons = stats.active_neurons.max(spiked);
            stats.sim_time_ms = sim_time;

            // Compute reward signal based on output activity
            self.reward_output_spike_count = output;
            if output >= self.reward_threshold {
                self.reward_signal = (self.reward_signal + 1.0).min(1.0);
            }
            stats.weight_updates = spiked;

            // Update firing rates (low-pass filter)
            let n = self.firing_rates.len();
            for (i, rate) in self.firing_rates.iter_mut().enumerate() {
                let spiked_here = local_spiking.contains(&i);
                let inst_freq = if spiked_here { 1000.0 / dt } else { 0.0 };
                *rate += (inst_freq - *rate) * dt / 100.0;
            }

            stats.mean_firing_rate = if n > 0 {
                self.firing_rates.iter().sum::<f64>() / n as f64
            } else { 0.0 };

            // Synchrony index: fraction of neurons firing in the most active step
            if let Some(hist) = self.spike_history.last()
                && !hist.is_empty() {
                    stats.synchrony_index = hist.iter().filter(|&&x| x).count() as f64 / hist.len() as f64;
                }

            // Weight statistics
            {
                let net = self.network.read();
                let syns = &net.synapses;
                if !syns.is_empty() {
                    let mean = syns.iter().map(|s| s.weight).sum::<f64>() / syns.len() as f64;
                    let var = syns.iter().map(|s| (s.weight - mean).powi(2)).sum::<f64>() / syns.len() as f64;
                    stats.weight_mean = mean;
                    stats.weight_std = var.sqrt();
                }
            }
        }

        self.step_count += 1;

        StepResult {
            spike_count: spike_count.load(Ordering::Relaxed),
            output_spike_count: output_spike_count.load(Ordering::Relaxed),
            spiking_neurons: local_spiking,
            output_spiking_neurons: local_output_spiking,
        }
    }

    fn step_neuron(i: usize, state: &mut NeuronState, dt: f64,
                   input_current: f64, model_params: &[NeuronModelParams]) -> bool {
        // Gating state is now stored in NeuronState, so all models
        // work as zero-sized types via the trait.
        match &model_params[i] {
            NeuronModelParams::Lif { .. } => {
                LifNeuron.step(state, dt, input_current)
            }
            NeuronModelParams::Izhikevich { .. } => {
                IzhikevichNeuron.step(state, dt, input_current)
            }
            NeuronModelParams::HodgkinHuxley { .. } => {
                HodgkinHuxleyNeuron.step(state, dt, input_current)
            }
        }
    }

    #[cfg(feature = "gpu")]
    fn step_gpu(&mut self) -> StepResult {
        let dt = self.dt;
        let total_spike_count = Arc::new(AtomicU64::new(0));
        let total_output_spike_count = Arc::new(AtomicU64::new(0));
        let mut local_spiking = Vec::new();
        let mut local_output_spiking = Vec::new();

        let input_f32: Vec<f32>;
        let sim_time: f64;

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

        let gpu = self.gpu_backend.as_mut().unwrap();
        gpu.upload_input_current(&input_f32);
        gpu.write_uniforms(dt as f32, self.network.read().time as f32);
        let (spiked, mem_pot, rec_var, refr_ctr, spike_ct) = gpu.step_lif();
        let gpu_input_current = gpu.step_spmv();

        {
            let mut net = self.network.write();
            let n = net.neuron_count();

            for i in 0..n {
                net.neurons.membrane_potential[i] = mem_pot[i] as f64;
                net.neurons.recovery_variable[i] = rec_var[i] as f64;
                net.neurons.refractory_counter[i] = refr_ctr[i];
                net.neurons.spike_count[i] = spike_ct[i] as u64;
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

    /// Get LFP estimate (population aggregate)
    pub fn lfp(&self) -> f64 {
        self.lfp_signal
    }

    /// Get firing rate for a specific neuron
    pub fn firing_rate(&self, neuron: usize) -> f64 {
        if neuron < self.firing_rates.len() {
            self.firing_rates[neuron]
        } else { 0.0 }
    }

    /// Get mean firing rate across all neurons
    pub fn mean_firing_rate(&self) -> f64 {
        if self.firing_rates.is_empty() { return 0.0; }
        self.firing_rates.iter().sum::<f64>() / self.firing_rates.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synapse::plasticity::StdpTrace;

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

    #[test]
    fn test_engine_sttdp_tracking() {
        let mut network = Network::new(50);
        let mut rng = StdRng::seed_from_u64(42);
        network.connect_random(0.1, &mut rng);
        // Enable plasticity on some synapses
        for syn in network.synapses.iter_mut() {
            syn.plasticity_enabled = true;
            syn.stdp_trace = Some(StdpTrace::new());
        }
        let mut engine = SimulationEngine::new(network)
            .with_noise(15.0);
        engine.simulate_ms(200.0);
        let stats = engine.stats();
        assert!(stats.total_spikes > 0);
    }

    #[test]
    fn test_conductance_dynamics() {
        let mut network = Network::new(100);
        let mut rng = StdRng::seed_from_u64(42);
        network.connect_random(0.05, &mut rng);
        let mut engine = SimulationEngine::new(network)
            .with_noise(8.0)
            .with_conductance(true);
        engine.simulate_ms(50.0);
        let stats = engine.stats();
        assert!(stats.sim_time_ms >= 0.0);
    }

    #[test]
    fn test_mean_firing_rate() {
        let mut network = Network::new(100);
        let mut rng = StdRng::seed_from_u64(42);
        network.connect_random(0.05, &mut rng);
        let mut engine = SimulationEngine::new(network).with_noise(15.0);
        engine.simulate_ms(200.0);
        let rate = engine.mean_firing_rate();
        assert!(rate >= 0.0);
    }
}

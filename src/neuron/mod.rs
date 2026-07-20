//! Neuron models and SoA storage.
//!
//! Three biologically-motivated models are provided:
//! - [`lif::LifNeuron`] — Leaky Integrate-and-Fire, fastest, medium fidelity
//! - [`izhikevich::IzhikevichNeuron`] — Izhikevich model, good speed/realism tradeoff
//! - [`hodgkin_huxley::HodgkinHuxleyNeuron`] — full conductance-based HH model
//!
//! All models implement the [`NeuronModel`] trait and operate on [`NeuronState`].
//! The canonical storage is [`NeuronArray`] (Struct of Arrays) for cache-friendly
//! bulk updates in the simulation engine.

pub mod hodgkin_huxley;
pub mod izhikevich;
pub mod lif;

use serde::{Deserialize, Serialize};

pub type NeuronId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
}

/// Per-neuron state stored in SoA (Struct of Arrays) layout
/// for cache-friendly bulk updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub membrane_potential: f64,
    pub recovery_variable: f64,
    pub refractory_counter: i32,
    pub last_spike_time: f64,
    pub spike_count: u64,
    pub neuron_type: NeuronType,
    pub model_params: NeuronModelParams,
    /// Hodgkin-Huxley gating variables (m, h, n)
    pub hh_m: f64,
    pub hh_h: f64,
    pub hh_n: f64,
    /// Set to true when neuron fires a spike in the current step
    pub just_spiked: bool,
}

impl NeuronState {
    pub fn new(neuron_type: NeuronType, params: NeuronModelParams) -> Self {
        Self {
            membrane_potential: params.resting_potential(),
            recovery_variable: 0.0,
            refractory_counter: 0,
            last_spike_time: -1e9,
            spike_count: 0,
            neuron_type,
            model_params: params,
            hh_m: 0.05,
            hh_h: 0.6,
            hh_n: 0.32,
            just_spiked: false,
        }
    }

    pub fn is_refractory(&self) -> bool {
        self.refractory_counter > 0
    }

    pub fn just_spiked(&self) -> bool {
        self.just_spiked
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum NeuronModelParams {
    /// Leaky Integrate-and-Fire: { resting, threshold, reset, tau_m, refractory_period }
    Lif {
        resting: f64,
        threshold: f64,
        reset: f64,
        tau_m: f64,
        refractory_period: f64,
        input_resistance: f64,
    },
    /// Izhikevich: { a, b, c, d }
    Izhikevich { a: f64, b: f64, c: f64, d: f64 },
    /// Hodgkin-Huxley: full conductance-based model params
    HodgkinHuxley {
        g_na: f64,
        g_k: f64,
        g_l: f64,
        e_na: f64,
        e_k: f64,
        e_l: f64,
        c_m: f64,
    },
}

impl NeuronModelParams {
    pub fn resting_potential(&self) -> f64 {
        match self {
            Self::Lif { resting, .. } => *resting,
            Self::Izhikevich { c, .. } => *c,
            Self::HodgkinHuxley { e_l, .. } => *e_l,
        }
    }
}

impl Default for NeuronModelParams {
    fn default() -> Self {
        Self::Lif {
            resting: -60.0,
            threshold: -50.0,
            reset: -65.0,
            tau_m: 8.0,
            refractory_period: 1.5,
            input_resistance: 2.0,
        }
    }
}

pub trait NeuronModel: Send + Sync {
    fn step(&mut self, state: &mut NeuronState, dt: f64, input_current: f64) -> bool;
    fn reset_state(&self, neuron_type: NeuronType) -> NeuronState;
}

/// SoA (Struct of Arrays) layout for bulk neuron updates.
/// This is the primary storage format for the simulation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronArray {
    pub membrane_potential: Vec<f64>,
    pub recovery_variable: Vec<f64>,
    pub refractory_counter: Vec<i32>,
    pub last_spike_time: Vec<f64>,
    pub spike_count: Vec<u64>,
    pub neuron_type: Vec<NeuronType>,
    pub model_params: Vec<NeuronModelParams>,
    pub input_current: Vec<f64>,
    pub is_output: Vec<bool>,
    /// Hodgkin-Huxley gating variables
    pub hh_m: Vec<f64>,
    pub hh_h: Vec<f64>,
    pub hh_n: Vec<f64>,
    /// Whether neuron fired in the most recent step
    pub just_spiked: Vec<bool>,
}

impl NeuronArray {
    pub fn new(size: usize) -> Self {
        let default_params = NeuronModelParams::default();
        Self {
            membrane_potential: vec![default_params.resting_potential(); size],
            recovery_variable: vec![0.0; size],
            refractory_counter: vec![0i32; size],
            last_spike_time: vec![-1e9; size],
            spike_count: vec![0u64; size],
            neuron_type: vec![NeuronType::Excitatory; size],
            model_params: vec![default_params; size],
            input_current: vec![0.0; size],
            is_output: vec![false; size],
            hh_m: vec![0.05; size],
            hh_h: vec![0.6; size],
            hh_n: vec![0.32; size],
            just_spiked: vec![false; size],
        }
    }

    pub fn len(&self) -> usize {
        self.membrane_potential.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
